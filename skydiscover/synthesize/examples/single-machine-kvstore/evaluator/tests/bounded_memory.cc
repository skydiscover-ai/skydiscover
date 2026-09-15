// Bounded memory under churn: RSS stays near the budget while far more distinct data flows through.
// Catches a store that never evicts or leaks, so memory grows until the process is killed.
//
// 256 MB budget. Churn ~2 GB of distinct data via rounds of insert-fresh + delete-old, keeping the live
// set within budget. Sample VmRSS each round; require peak RSS to stay well under 512 MB.
#include "kvstore_interface.h"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <string>

static size_t read_vmrss_kb() {
    FILE* f = fopen("/proc/self/status", "r");
    if (!f) return 0;
    char line[256];
    size_t kb = 0;
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "VmRSS:", 6) == 0) { sscanf(line + 6, "%zu", &kb); break; }
    }
    fclose(f);
    return kb;
}

int main(int argc, char** argv) {
    std::string path = argc > 1 ? argv[1] : "/tmp/check_bounded_memory";
    std::string q = "\x27" + path + "\x27";
    system(("rm -rf " + q + " 2>/dev/null; mkdir -p " + q).c_str());  // impl-agnostic clean slate

    IKVStore* store = create_kvstore();
    const size_t kBudget = 256ull * 1024 * 1024;  // 256 MB
    store->InitExtended(1 << 20, 0, kBudget, path.c_str());
    store->StartSession();

    const size_t kValSize = 4096;
    const size_t kLiveKeys = (180ull * 1024 * 1024) / kValSize;  // ~46k keys live (~180MB, within budget)
    const int kRounds = 12;                                      // ~2.2 GB churned

    GenValue v; v.size = (uint32_t)kValSize;
    size_t peak_kb = 0; uint64_t next_key = 0; uint64_t churned = 0;

    for (int round = 0; round < kRounds; ++round) {
        uint64_t round_start = next_key;
        for (size_t i = 0; i < kLiveKeys; ++i) {
            uint64_t key = next_key++;
            memcpy(v.data, &key, 8);
            memset(v.data + 8, (int)(key & 0xff), 32);
            store->Upsert(key, v);
            churned += kValSize;
        }
        if (round > 0) {
            uint64_t del_start = round_start - kLiveKeys;
            for (size_t i = 0; i < kLiveKeys; ++i) store->Delete(del_start + i);
        }
        store->Refresh();
        size_t rss = read_vmrss_kb();
        if (rss > peak_kb) peak_kb = rss;
    }
    store->StopSession();
    system(("rm -rf " + q + " 2>/dev/null").c_str());

    double peak_mb = peak_kb / 1024.0, churned_gb = churned / 1e9;
    bool pass = (peak_kb < (size_t)(512ull * 1024)) && (churned_gb >= 1.5);
    printf("BOUNDED_MEMORY peak_RSS=%.0f MB churned=%.2f GB budget=256 MB -> %s\n",
           peak_mb, churned_gb, pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
