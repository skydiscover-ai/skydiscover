// initial_program.cc — trivial in-memory seed for the kvstore benchmark family.
//
// This is a correct-but-unoptimized starting point: a sharded hash table guarded
// by per-shard mutexes, with synchronous reads. A search strategy (jitskit,
// claude_code, or an evolutionary loop) is expected to evolve the region between
// the EVOLVE-BLOCK markers into a specialized, high-throughput design.
//
// It implements the full IKVStore contract from the shared harness. Build it
// against benchmarks/kvstore/_harness/ (populated from the runtime — see the
// family README).

#include "../_harness/kvstore_interface.h"

#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

// EVOLVE-BLOCK-START
//
// Everything between these markers is the design surface. The seed is a fixed
// fan-out of std::unordered_map shards, each behind its own mutex. Reads,
// upserts, RMWs, and deletes hash the key to a shard and take that shard's lock.
// There is no disk tier, no eviction, and no async I/O — reads complete
// synchronously. Replace this with whatever structure best fits the spec.

namespace {

constexpr size_t kNumShards = 256;  // power of two, so (key & mask) selects a shard

class SeedKVStore : public IKVStore {
public:
    void Init(size_t /*hash_table_size*/, size_t /*log_size_bytes*/) override {
        // Nothing to pre-size: the shard maps grow on demand.
    }

    void StartSession() override {}
    void StopSession() override {}
    void Refresh() override {}

    bool Read(uint64_t key, GenValue& out) override {
        Shard& s = shard_for(key);
        std::lock_guard<std::mutex> lock(s.mu);
        auto it = s.map.find(key);
        if (it == s.map.end()) return false;
        out = it->second;
        return true;
    }

    // Synchronous store: ReadAsync completes immediately (verbatim wrapper from
    // the interface header) and CompletePending is a no-op.
    OpStatus ReadAsync(ReadSlot* slot) override {
        bool found = Read(slot->key, slot->out);
        slot->status = found ? OpStatus::Ok : OpStatus::NotFound;
        slot->done.store(1, std::memory_order_release);
        return slot->status;
    }

    void CompletePending(bool /*wait*/) override {}

    void Upsert(uint64_t key, const GenValue& value) override {
        Shard& s = shard_for(key);
        std::lock_guard<std::mutex> lock(s.mu);
        s.map[key] = value;
    }

    // RMW interpreted as an atomic integer add on the leading 8 bytes of the
    // value (the YCSB-style counter semantic). Absent key => initialize from
    // the modification.
    void RMW(uint64_t key, const uint8_t* mod_data, size_t mod_size) override {
        uint64_t delta = 0;
        std::memcpy(&delta, mod_data, mod_size < sizeof(delta) ? mod_size : sizeof(delta));
        Shard& s = shard_for(key);
        std::lock_guard<std::mutex> lock(s.mu);
        GenValue& v = s.map[key];
        uint64_t cur = 0;
        if (v.size >= sizeof(cur)) std::memcpy(&cur, v.data, sizeof(cur));
        cur += delta;
        std::memcpy(v.data, &cur, sizeof(cur));
        if (v.size < sizeof(cur)) v.size = sizeof(cur);
    }

    bool Delete(uint64_t key) override {
        Shard& s = shard_for(key);
        std::lock_guard<std::mutex> lock(s.mu);
        return s.map.erase(key) > 0;
    }

    void Checkpoint() override {}  // in-memory: no durable state to persist

private:
    struct Shard {
        std::mutex mu;
        std::unordered_map<uint64_t, GenValue> map;
    };

    Shard& shard_for(uint64_t key) { return shards_[key & (kNumShards - 1)]; }

    std::vector<Shard> shards_{kNumShards};
};

}  // namespace

IKVStore* create_kvstore() { return new SeedKVStore(); }
//
// EVOLVE-BLOCK-END
