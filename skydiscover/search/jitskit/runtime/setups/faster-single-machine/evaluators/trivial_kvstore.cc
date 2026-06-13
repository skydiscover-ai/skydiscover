// trivial_kvstore.cc — Reference correct KV store for unit tests.
// Uses std::unordered_map + std::mutex. Deliberately simple and slow
// but guaranteed correct. Used to verify the harnesses themselves work.

#include "../../../interface/kvstore_interface.h"
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>

#include <cerrno>
#include <cstdio>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

class TrivialStore : public IKVStore {
public:
    TrivialStore() = default;
    ~TrivialStore() override = default;

    void Init(size_t hash_table_size, size_t log_size_bytes) override {
        InitExtended(hash_table_size, log_size_bytes, 0, nullptr);
    }

    void InitExtended(size_t /*hash_table_size*/, size_t /*log_size_bytes*/,
                      size_t /*mem_budget_bytes*/, const char* storage_path) override {
        std::lock_guard<std::mutex> lk(mu_);
        map_.clear();
        snapshot_path_.clear();
        if (storage_path == nullptr || storage_path[0] == '\0') {
            return;
        }

        storage_dir_ = storage_path;
        if (::mkdir(storage_dir_.c_str(), 0755) != 0 && errno != EEXIST) {
            std::perror("mkdir");
            throw std::runtime_error("failed to create checkpoint directory");
        }
        snapshot_path_ = storage_dir_ + "/trivial_store.snapshot";

        FILE* f = std::fopen(snapshot_path_.c_str(), "rb");
        if (!f) {
            return;
        }

        uint64_t entry_count = 0;
        if (std::fread(&entry_count, sizeof(entry_count), 1, f) != 1) {
            std::fclose(f);
            throw std::runtime_error("failed to read checkpoint header");
        }

        for (uint64_t i = 0; i < entry_count; ++i) {
            uint64_t key = 0;
            GenValue value{};
            if (std::fread(&key, sizeof(key), 1, f) != 1 ||
                std::fread(&value, sizeof(value), 1, f) != 1) {
                std::fclose(f);
                throw std::runtime_error("failed to read checkpoint entry");
            }
            map_[key] = value;
        }

        std::fclose(f);
    }

    void StartSession() override {}
    void StopSession() override {}
    void Refresh() override {}

    bool Read(uint64_t key, GenValue& out) override {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = map_.find(key);
        if (it == map_.end()) return false;
        out = it->second;
        return true;
    }

    void Upsert(uint64_t key, const GenValue& value) override {
        std::lock_guard<std::mutex> lk(mu_);
        map_[key] = value;
    }

    bool Delete(uint64_t key) override {
        std::lock_guard<std::mutex> lk(mu_);
        return map_.erase(key) > 0;
    }

    void RMW(uint64_t key, const uint8_t* mod_data, size_t mod_size) override {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = map_.find(key);
        uint64_t cur = 0;
        if (it != map_.end() && it->second.size >= 8) {
            std::memcpy(&cur, it->second.data, 8);
        }
        uint64_t delta = 0;
        size_t n = mod_size < 8 ? mod_size : 8;
        std::memcpy(&delta, mod_data, n);
        uint64_t result = cur + delta;
        GenValue v{};
        v.size = 8;
        std::memcpy(v.data, &result, 8);
        map_[key] = v;
    }

    void Checkpoint() override {
        std::lock_guard<std::mutex> lk(mu_);
        if (snapshot_path_.empty()) {
            return;
        }

        const std::string temp_path = snapshot_path_ + ".tmp";
        FILE* f = std::fopen(temp_path.c_str(), "wb");
        if (!f) {
            throw std::runtime_error("failed to open checkpoint temp file");
        }

        const uint64_t entry_count = map_.size();
        bool ok = std::fwrite(&entry_count, sizeof(entry_count), 1, f) == 1;
        for (const auto& [key, value] : map_) {
            if (!ok) {
                break;
            }
            ok = std::fwrite(&key, sizeof(key), 1, f) == 1 &&
                 std::fwrite(&value, sizeof(value), 1, f) == 1;
        }

        ok = ok && std::fflush(f) == 0 && ::fsync(::fileno(f)) == 0;
        std::fclose(f);

        if (!ok || ::rename(temp_path.c_str(), snapshot_path_.c_str()) != 0) {
            ::unlink(temp_path.c_str());
            throw std::runtime_error("failed to finalize checkpoint");
        }
    }

private:
    std::mutex mu_;
    std::unordered_map<uint64_t, GenValue> map_;
    std::string storage_dir_;
    std::string snapshot_path_;
};

IKVStore* create_kvstore() {
    return new TrivialStore();
}
