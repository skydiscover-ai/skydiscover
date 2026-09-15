// reference_kvstore.cc
//
// A CORRECT, slow, trusted reference KV-store implementation, intended to be
// used as a PASS-oracle that validates tests against a fast candidate
// store.  Correctness is the ONLY design goal here.  Performance is explicitly
// a non-goal: the implementation favours simple, obviously-correct strategies
// over fast ones everywhere there is a trade-off.
//
// The four correctness properties this file guarantees:
//
//   1. VALUE FIDELITY.  Read returns the exact bytes (and size) most recently
//      Upserted/RMW'd for a key.  Absent or deleted keys return false.  Stored
//      values are the literal bytes the caller handed us; they are never
//      regenerated from the key.
//
//   2. BOUNDED MEMORY.  We respect mem_budget_bytes.  When the live working set
//      exceeds the budget we evict the coldest entries' payloads to an on-disk
//      spill file (LRU) and reload them on demand.  Process RSS therefore stays
//      near the budget rather than growing without bound.  The on-disk copy is
//      the source of truth for evicted entries.
//
//   3. DURABILITY ACROSS A HARD CRASH.  Checkpoint() atomically persists the
//      COMPLETE current state (every live key's exact bytes AND the set of
//      tombstoned/deleted keys) using write-to-temp + fsync + rename + fsync of
//      the directory.  After a SIGKILL following a successful Checkpoint(), a
//      fresh instance pointed at the same storage_path recovers EXACTLY the
//      checkpointed state.
//
//   4. NO SILENT LOSS ON I/O ERROR.  Every pwrite/pread/write/read/fsync return
//      value is checked.  On an unrecoverable I/O error we print a diagnostic
//      and abort the process (nonzero exit) rather than acknowledging a write we
//      failed to persist.
//
// Implementation strategy:
//   * One global std::mutex guards all state.  This serialises every operation,
//     which is the simplest way to be unambiguously correct under the interface
//     contract ("all operations except Init/StartSession/StopSession must be
//     thread-safe").  Slow but trusted, exactly what an oracle wants.
//   * Each live key maps to an Entry that holds either the value bytes resident
//     in memory, OR an offset into the on-disk spill file if it has been evicted
//     to stay within the memory budget.
//   * An LRU list orders resident entries by recency so we always evict the
//     coldest payload first.
//
// Build: g++ -std=c++17 -O2 -I<dir-with-header> harness.cc reference_kvstore.cc -lpthread -o h

#include "kvstore_interface.h"

#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <list>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace {

// ── Fatal-error helper ───────────────────────────────────────────────────────
// Used for any unrecoverable I/O failure.  A correct store must NOT acknowledge
// an operation it could not durably perform and then silently lose it, so we
// abort the whole process loudly instead.
[[noreturn]] void fatal(const char* what) {
    int e = errno;
    std::fprintf(stderr, "reference_kvstore FATAL: %s: %s\n", what,
                 e ? std::strerror(e) : "(no errno)");
    std::fflush(stderr);
    std::abort();
}

// ── Robust POSIX I/O wrappers ────────────────────────────────────────────────
// Loop over short reads/writes and treat any genuine failure as fatal.

void write_all_at(int fd, const void* buf, size_t len, off_t off) {
    const uint8_t* p = static_cast<const uint8_t*>(buf);
    size_t done = 0;
    while (done < len) {
        ssize_t n = ::pwrite(fd, p + done, len - done, off + static_cast<off_t>(done));
        if (n < 0) {
            if (errno == EINTR) continue;
            fatal("pwrite");
        }
        if (n == 0) fatal("pwrite returned 0 (no progress)");
        done += static_cast<size_t>(n);
    }
}

void read_all_at(int fd, void* buf, size_t len, off_t off) {
    uint8_t* p = static_cast<uint8_t*>(buf);
    size_t done = 0;
    while (done < len) {
        ssize_t n = ::pread(fd, p + done, len - done, off + static_cast<off_t>(done));
        if (n < 0) {
            if (errno == EINTR) continue;
            fatal("pread");
        }
        if (n == 0) fatal("pread hit EOF before satisfying request (corrupt/short file)");
        done += static_cast<size_t>(n);
    }
}

void write_all(int fd, const void* buf, size_t len) {
    const uint8_t* p = static_cast<const uint8_t*>(buf);
    size_t done = 0;
    while (done < len) {
        ssize_t n = ::write(fd, p + done, len - done);
        if (n < 0) {
            if (errno == EINTR) continue;
            fatal("write");
        }
        if (n == 0) fatal("write returned 0 (no progress)");
        done += static_cast<size_t>(n);
    }
}

void read_all(int fd, void* buf, size_t len) {
    uint8_t* p = static_cast<uint8_t*>(buf);
    size_t done = 0;
    while (done < len) {
        ssize_t n = ::read(fd, p + done, len - done);
        if (n < 0) {
            if (errno == EINTR) continue;
            fatal("read");
        }
        if (n == 0) fatal("read hit EOF before satisfying request (corrupt/short file)");
        done += static_cast<size_t>(n);
    }
}

void fsync_or_die(int fd) {
    while (::fsync(fd) != 0) {
        if (errno == EINTR) continue;
        fatal("fsync");
    }
}

// fsync the directory containing `path` so that a rename into it is durable.
void fsync_dir(const std::string& dir) {
    int dfd = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY);
    if (dfd < 0) fatal("open(dir) for fsync");
    fsync_or_die(dfd);
    ::close(dfd);
}

// ── Reference store ──────────────────────────────────────────────────────────
class ReferenceKVStore final : public IKVStore {
public:
    ReferenceKVStore() = default;
    ~ReferenceKVStore() override {
        if (spill_fd_ >= 0) ::close(spill_fd_);
    }

    // ── Lifecycle ────────────────────────────────────────────────────────────

    void Init(size_t hash_table_size, size_t log_size_bytes) override {
        // No storage path / budget: in-memory, unbounded.  We still honour the
        // full interface for completeness.
        InitExtended(hash_table_size, log_size_bytes, /*mem_budget=*/0,
                     /*storage_path=*/nullptr);
    }

    void InitExtended(size_t hash_table_size, size_t /*log_size_bytes*/,
                      size_t mem_budget_bytes, const char* storage_path) override {
        mem_budget_ = mem_budget_bytes;  // 0 == unlimited
        if (storage_path && *storage_path) {
            storage_dir_ = storage_path;
            // Ensure the directory exists (idempotent).
            ::mkdir(storage_dir_.c_str(), 0755);
            checkpoint_path_ = storage_dir_ + "/refkv.checkpoint";
            checkpoint_tmp_  = storage_dir_ + "/refkv.checkpoint.tmp";
            spill_path_      = storage_dir_ + "/refkv.spill";
            persistent_ = true;
        }

        if (hash_table_size) map_.reserve(hash_table_size);

        // Open (truncate) a fresh spill file used purely as an eviction backing
        // store for the CURRENT process lifetime.  It is NOT the durable
        // checkpoint, Checkpoint() writes its own self-contained file.  We
        // truncate because spilled offsets are only meaningful within one run.
        if (persistent_) {
            open_spill_file(/*truncate=*/true);
        }

        // Recover any previously checkpointed state.  Recovery is performed in
        // Refresh()/StartSession too, but doing it here makes a bare
        // InitExtended+Read also see the data; it is idempotent.
        recover_from_checkpoint();
    }

    // Session and maintenance hooks.  Recovery is idempotent so calling it from
    // Refresh()/StartSession is harmless; it makes the "InitExtended +
    // StartSession + Refresh" recovery sequence in the spec robust regardless of
    // which call the caller relies on.
    void StartSession() override {
        std::lock_guard<std::mutex> lk(mu_);
        recover_from_checkpoint_locked();
    }
    void StopSession() override {}
    void Refresh() override {
        std::lock_guard<std::mutex> lk(mu_);
        recover_from_checkpoint_locked();
    }

    // ── Reads ────────────────────────────────────────────────────────────────

    bool Read(uint64_t key, GenValue& out) override {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = map_.find(key);
        if (it == map_.end()) return false;  // absent or deleted
        load_value_locked(key, it->second, out);  // also marks MRU / faults in
        return true;
    }

    OpStatus ReadAsync(ReadSlot* slot) override {
        // Synchronous wrapper, exactly as the header prescribes for stores
        // without a non-blocking I/O path.
        bool found = Read(slot->key, slot->out);
        slot->status = found ? OpStatus::Ok : OpStatus::NotFound;
        slot->done.store(1, std::memory_order_release);
        return slot->status;
    }

    void CompletePending(bool /*wait*/) override {
        // All reads complete synchronously, so there is nothing to drive.
    }

    // ── Writes ─────────────────────────────────────────────────────────────────

    void Upsert(uint64_t key, const GenValue& value) override {
        std::lock_guard<std::mutex> lk(mu_);
        uint32_t sz = value.size;
        if (sz > GenValue::kMaxSize) fatal("Upsert value.size exceeds kMaxSize");
        store_value_locked(key, value.data, sz);
    }

    void RMW(uint64_t key, const uint8_t* mod_data, size_t mod_size) override {
        // RMW semantic (matches the harnesses' FASTER-style modification):
        //   * If the key exists, interpret the first 8 bytes of the stored value
        //     and of mod_data as little-endian uint64 counters and add them,
        //     leaving the value's size and all other bytes untouched.
        //   * If the key is absent, insert a new value equal to mod_data[0..mod_size)
        //     (the modification IS the initial value), with size = mod_size.
        // Done entirely under the global lock, so concurrent RMWs cannot lose
        // updates.
        std::lock_guard<std::mutex> lk(mu_);
        if (mod_size > GenValue::kMaxSize) fatal("RMW mod_size exceeds kMaxSize");

        auto it = map_.find(key);
        if (it == map_.end()) {
            // Insert initial value = the modification bytes.
            store_value_locked(key, mod_data, static_cast<uint32_t>(mod_size));
            return;
        }

        // Existing key: load current bytes, apply the counter add, store back.
        GenValue cur;
        load_value_locked(key, it->second, cur);

        uint64_t cur_ctr = 0, mod_ctr = 0;
        std::memcpy(&cur_ctr, cur.data, sizeof(uint64_t) <= cur.size ? sizeof(uint64_t) : cur.size);
        std::memcpy(&mod_ctr, mod_data, sizeof(uint64_t) <= mod_size ? sizeof(uint64_t) : mod_size);
        uint64_t new_ctr = cur_ctr + mod_ctr;
        std::memcpy(cur.data, &new_ctr, sizeof(uint64_t) <= cur.size ? sizeof(uint64_t) : cur.size);

        store_value_locked(key, cur.data, cur.size);
    }

    bool Delete(uint64_t key) override {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = map_.find(key);
        if (it == map_.end()) return false;  // no-op if absent
        evict_accounting_remove_locked(it->second);
        map_.erase(it);
        // Record the tombstone so a Checkpoint() persists the delete and a
        // recovery after crash keeps the key absent.
        deleted_.insert(key);
        return true;
    }

    // ── Durable checkpoint ───────────────────────────────────────────────────
    //
    // File format (self-describing, fixed-endian on this host which is fine for
    // an oracle that always runs on the same machine):
    //
    //   [ magic:u64 ]
    //   [ version:u64 ]
    //   [ live_count:u64 ]
    //   [ deleted_count:u64 ]
    //   live records, each: [ key:u64 ][ size:u32 ][ size bytes ]
    //   deleted keys, each: [ key:u64 ]
    //
    // Written to a temp file, fsync'd, atomically renamed over the real file,
    // and the directory is fsync'd so the rename itself is durable.
    void Checkpoint() override {
        std::lock_guard<std::mutex> lk(mu_);
        if (!persistent_) return;  // nothing to persist for an in-memory store

        int fd = ::open(checkpoint_tmp_.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd < 0) fatal("open(checkpoint tmp)");

        const uint64_t magic = kMagic;
        const uint64_t version = kVersion;
        const uint64_t live_count = map_.size();
        const uint64_t del_count = deleted_.size();

        write_all(fd, &magic, sizeof(magic));
        write_all(fd, &version, sizeof(version));
        write_all(fd, &live_count, sizeof(live_count));
        write_all(fd, &del_count, sizeof(del_count));

        // Live records.  We materialise each value (it may currently live on the
        // spill file) and write the EXACT bytes.
        GenValue tmp;
        for (auto& kv : map_) {
            uint64_t key = kv.first;
            load_value_locked(key, kv.second, tmp);
            uint32_t sz = tmp.size;
            write_all(fd, &key, sizeof(key));
            write_all(fd, &sz, sizeof(sz));
            if (sz) write_all(fd, tmp.data, sz);
        }

        // Deleted keys (tombstones).
        for (uint64_t k : deleted_) {
            write_all(fd, &k, sizeof(k));
        }

        fsync_or_die(fd);
        if (::close(fd) != 0) fatal("close(checkpoint tmp)");

        if (::rename(checkpoint_tmp_.c_str(), checkpoint_path_.c_str()) != 0)
            fatal("rename(checkpoint)");

        // Make the rename itself durable.
        fsync_dir(storage_dir_);
    }

    CacheStats GetCacheStats() const override {
        std::lock_guard<std::mutex> lk(mu_);
        CacheStats cs;
        cs.hot_bytes = resident_bytes_;
        cs.total_bytes = total_value_bytes_;
        cs.budget_bytes = mem_budget_;
        cs.evictions = evictions_;
        return cs;
    }

private:
    // ── Per-key entry ──────────────────────────────────────────────────────────
    // An entry's value is in exactly one of two places:
    //   * resident == true : `bytes` holds the value, and `lru_it` points at its
    //     node in the LRU recency list.
    //   * resident == false: the value lives in the spill file at [spill_off,
    //     spill_off+size).  `bytes` is empty.
    struct Entry {
        uint32_t size = 0;
        bool resident = true;
        std::vector<uint8_t> bytes;          // valid iff resident
        off_t spill_off = 0;                 // valid iff !resident
        std::list<uint64_t>::iterator lru_it; // valid iff resident
    };

    // ── Eviction accounting helpers (all called under mu_) ─────────────────────

    // Remove an entry's contribution to resident/total byte counters and, if it
    // is resident, unlink it from the LRU list.  Used on Delete and overwrite.
    void evict_accounting_remove_locked(Entry& e) {
        total_value_bytes_ -= e.size;
        if (e.resident) {
            resident_bytes_ -= e.size;
            lru_.erase(e.lru_it);
        }
    }

    // Insert or overwrite key->value with the given bytes, keeping all counters
    // and the LRU list consistent, then enforce the memory budget.
    void store_value_locked(uint64_t key, const uint8_t* data, uint32_t size) {
        auto it = map_.find(key);
        if (it != map_.end()) {
            // Overwrite: drop the old accounting first.
            evict_accounting_remove_locked(it->second);
            Entry& e = it->second;
            e.size = size;
            e.resident = true;
            e.bytes.assign(data, data + size);
            lru_.push_back(key);
            e.lru_it = std::prev(lru_.end());
            resident_bytes_ += size;
            total_value_bytes_ += size;
        } else {
            // Fresh insert.  This key may previously have been deleted; clearing
            // the tombstone keeps the live/deleted sets disjoint.
            deleted_.erase(key);
            Entry e;
            e.size = size;
            e.resident = true;
            e.bytes.assign(data, data + size);
            lru_.push_back(key);
            e.lru_it = std::prev(lru_.end());
            resident_bytes_ += size;
            total_value_bytes_ += size;
            map_.emplace(key, std::move(e));
        }
        enforce_budget_locked();
    }

    // Fill `out` with the value for a (possibly evicted) entry, faulting it back
    // in from the spill file if necessary.  Either way the entry ends up
    // resident and marked most-recently-used; if a fault-in pushed us over the
    // budget we re-enforce it.
    void load_value_locked(uint64_t key, Entry& e, GenValue& out) {
        out.size = e.size;
        if (e.resident) {
            if (e.size) std::memcpy(out.data, e.bytes.data(), e.size);
            // Mark MRU.
            lru_.erase(e.lru_it);
            lru_.push_back(key);
            e.lru_it = std::prev(lru_.end());
            return;
        }
        // Evicted: read the exact bytes back from the spill file (source of
        // truth for evicted entries).
        if (e.size) {
            read_all_at(spill_fd_, out.data, e.size, e.spill_off);
        }
        // Fault it back into memory as MRU.
        e.resident = true;
        e.bytes.assign(out.data, out.data + e.size);
        resident_bytes_ += e.size;
        lru_.push_back(key);
        e.lru_it = std::prev(lru_.end());
        enforce_budget_locked();
    }

    // Evict coldest resident entries to the spill file until resident_bytes_ is
    // within the budget.  mem_budget_ == 0 means unlimited (never evict).
    void enforce_budget_locked() {
        if (mem_budget_ == 0) return;
        if (!persistent_) return;  // nowhere to spill to; keep in memory
        while (resident_bytes_ > mem_budget_ && !lru_.empty()) {
            uint64_t victim_key = lru_.front();
            auto it = map_.find(victim_key);
            if (it == map_.end()) {
                // Stale LRU node (should not happen); drop and continue.
                lru_.pop_front();
                continue;
            }
            Entry& e = it->second;
            // Append the victim's bytes to the spill file.  Append-only growth
            // means previously-written offsets remain valid; we never rewrite.
            off_t off = spill_tail_;
            if (e.size) {
                write_all_at(spill_fd_, e.bytes.data(), e.size, off);
                // No fsync here: the spill file is process-local scratch, not the
                // durable checkpoint.  Correctness on crash comes from Checkpoint().
            }
            e.spill_off = off;
            spill_tail_ += e.size;
            e.resident = false;
            e.bytes.clear();
            e.bytes.shrink_to_fit();
            resident_bytes_ -= e.size;
            lru_.pop_front();
            ++evictions_;
        }
    }

    // ── Spill-file lifecycle ───────────────────────────────────────────────────
    void open_spill_file(bool truncate) {
        int flags = O_RDWR | O_CREAT;
        if (truncate) flags |= O_TRUNC;
        spill_fd_ = ::open(spill_path_.c_str(), flags, 0644);
        if (spill_fd_ < 0) fatal("open(spill)");
        spill_tail_ = 0;
    }

    // ── Recovery ───────────────────────────────────────────────────────────────
    void recover_from_checkpoint() {
        std::lock_guard<std::mutex> lk(mu_);
        recover_from_checkpoint_locked();
    }

    // Load the durable checkpoint into memory, replacing all current state.
    // Idempotent: guarded by recovered_ so repeated StartSession/Refresh calls
    // don't reload (and overwrite freshly-written data).
    void recover_from_checkpoint_locked() {
        if (!persistent_ || recovered_) return;
        recovered_ = true;

        int fd = ::open(checkpoint_path_.c_str(), O_RDONLY);
        if (fd < 0) {
            // No checkpoint yet: clean empty state.  Not an error.
            return;
        }

        uint64_t magic = 0, version = 0, live_count = 0, del_count = 0;
        read_all(fd, &magic, sizeof(magic));
        read_all(fd, &version, sizeof(version));
        if (magic != kMagic) fatal("checkpoint magic mismatch (corrupt file)");
        if (version != kVersion) fatal("checkpoint version mismatch");
        read_all(fd, &live_count, sizeof(live_count));
        read_all(fd, &del_count, sizeof(del_count));

        // Reset live state.
        map_.clear();
        deleted_.clear();
        lru_.clear();
        resident_bytes_ = 0;
        total_value_bytes_ = 0;

        // Live records.  We load them all in as resident, then enforce the
        // budget so a recovered store also stays within memory bounds.
        std::vector<uint8_t> buf;
        for (uint64_t i = 0; i < live_count; ++i) {
            uint64_t key = 0;
            uint32_t sz = 0;
            read_all(fd, &key, sizeof(key));
            read_all(fd, &sz, sizeof(sz));
            if (sz > GenValue::kMaxSize) fatal("checkpoint record size exceeds kMaxSize");
            buf.resize(sz);
            if (sz) read_all(fd, buf.data(), sz);
            // Insert directly (bypass store_value_locked's tombstone-clearing,
            // which is unnecessary on a clean reload).
            Entry e;
            e.size = sz;
            e.resident = true;
            e.bytes.assign(buf.begin(), buf.end());
            lru_.push_back(key);
            e.lru_it = std::prev(lru_.end());
            resident_bytes_ += sz;
            total_value_bytes_ += sz;
            map_.emplace(key, std::move(e));
        }

        // Deleted keys.
        for (uint64_t i = 0; i < del_count; ++i) {
            uint64_t k = 0;
            read_all(fd, &k, sizeof(k));
            deleted_.insert(k);
        }

        ::close(fd);
        enforce_budget_locked();
    }

    // ── State (all guarded by mu_) ─────────────────────────────────────────────
    mutable std::mutex mu_;

    std::unordered_map<uint64_t, Entry> map_;  // live keys -> value/location
    std::unordered_set<uint64_t> deleted_;     // tombstoned keys (for durability)
    std::list<uint64_t> lru_;                  // front = coldest, back = hottest

    size_t mem_budget_ = 0;       // 0 == unlimited
    size_t resident_bytes_ = 0;   // bytes currently held in memory
    size_t total_value_bytes_ = 0;// total live value bytes (mem + spill)
    uint64_t evictions_ = 0;

    bool persistent_ = false;
    bool recovered_ = false;
    std::string storage_dir_;
    std::string checkpoint_path_;
    std::string checkpoint_tmp_;
    std::string spill_path_;

    int spill_fd_ = -1;
    off_t spill_tail_ = 0;  // append offset into the spill file

    static constexpr uint64_t kMagic = 0x5245464b56303031ULL;  // "REFKV001"
    static constexpr uint64_t kVersion = 1;
};

}  // namespace

// ── Factory ──────────────────────────────────────────────────────────────────
IKVStore* create_kvstore() { return new ReferenceKVStore(); }
