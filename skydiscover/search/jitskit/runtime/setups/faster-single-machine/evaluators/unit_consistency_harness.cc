// unit_consistency_harness.cc — Small-scale crash recovery tests for unit tests.
// Tests FASTER's monotonicity property with 500 keys.
// Usage: ./unit_consistency <test_id> <num_threads> [storage_path]
//   test_id: 1=crash_recovery_fuzzy  2=crash_recovery_no_ckpt

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include "../../../interface/kvstore_interface.h"

static constexpr size_t kNumKeys = 500;
static constexpr size_t kRefreshInterval = 64;
static constexpr size_t kValueSize = 8;
static constexpr size_t kHashTableSize = 1024;
static constexpr size_t kLogSize = 1ULL << 20;
static constexpr uint64_t kInitialValue = 42;
static const uint8_t kModData[8] = {5, 0, 0, 0, 0, 0, 0, 0};

static IKVStore* g_store = nullptr;

static inline uint64_t decode_u64(const GenValue& v) {
    uint64_t val = 0;
    if (v.size >= 8) std::memcpy(&val, v.data, 8);
    return val;
}

struct PairRecord {
    uint64_t key_r1;
    uint64_t key_r2;
};

static void load_keys() {
    g_store->StartSession();
    GenValue val{}; val.size = kValueSize;
    std::memcpy(val.data, &kInitialValue, sizeof(kInitialValue));
    for (size_t i = 0; i < kNumKeys; i++) g_store->Upsert(i, val);
    g_store->StopSession();
}

// Shared verification: for every (r1, r2) pair issued in order by a thread,
// the recovered state must be {none, r1 only, both} — never r2 without r1.
static void verify_monotonicity(const char* test_name,
                                const std::vector<std::vector<PairRecord>>& thread_pairs,
                                int num_threads,
                                bool require_liveness) {
    g_store->StartSession();

    uint64_t pairs_checked = 0, violations = 0;
    uint64_t both_present = 0, r1_only = 0, neither = 0;

    for (int t = 0; t < num_threads; t++) {
        for (const auto& pair : thread_pairs[t]) {
            GenValue v1{}, v2{};
            bool has_r1 = g_store->Read(pair.key_r1, v1);
            bool has_r2 = g_store->Read(pair.key_r2, v2);

            uint64_t val_r1 = has_r1 ? decode_u64(v1) : 0;
            uint64_t val_r2 = has_r2 ? decode_u64(v2) : 0;

            bool r1_applied = has_r1 && (val_r1 > kInitialValue);
            bool r2_applied = has_r2 && (val_r2 > kInitialValue);

            if (r1_applied && r2_applied) both_present++;
            else if (r1_applied && !r2_applied) r1_only++;
            else if (!r1_applied && !r2_applied) neither++;
            else {
                violations++;
                if (violations <= 10) {
                    fprintf(stderr, "  VIOLATION: thread=%d key_r1=%lu (val=%lu) "
                            "key_r2=%lu (val=%lu) — r2 persisted without r1\n",
                            t, pair.key_r1, val_r1, pair.key_r2, val_r2);
                }
            }
            pairs_checked++;
        }
    }
    g_store->StopSession();

    bool liveness_failed = false;
    if (require_liveness && (both_present + r1_only) == 0 && pairs_checked > 0) {
        liveness_failed = true;
        fprintf(stderr, "  LIVENESS FAILURE: Checkpoint() was called but no writes "
                "survived recovery (%lu pairs, all 'neither').\n", pairs_checked);
    }

    printf("TEST %s pairs=%lu both=%lu r1_only=%lu neither=%lu violations=%lu liveness=%s\n",
           test_name, pairs_checked, both_present, r1_only, neither, violations,
           liveness_failed ? "FAILED" : "ok");

    if (violations == 0 && !liveness_failed)
        fprintf(stderr, "%s: PASSED (both=%lu r1_only=%lu neither=%lu)\n",
                test_name, both_present, r1_only, neither);
    else if (violations > 0)
        fprintf(stderr, "%s: FAILED (%lu violations)\n", test_name, violations);
    else
        fprintf(stderr, "%s: FAILED (liveness — Checkpoint() did not persist any data)\n",
                test_name);
}

// ── TEST 1: Fuzzy checkpoint crash recovery ─────────────────────────────────
// Writers do paired RMWs CONCURRENTLY with Checkpoint().
static void test_crash_recovery_fuzzy(int num_threads, const char* storage_path) {
    fprintf(stderr, "=== TEST 1: CRASH RECOVERY (Fuzzy Checkpoint, small scale) ===\n");

    load_keys();

    std::vector<std::vector<PairRecord>> thread_pairs(num_threads);
    std::atomic<bool> writers_running{true};
    std::atomic<size_t> pairs_started{0};

    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; t++) {
        threads.emplace_back([t, &thread_pairs, &writers_running, &pairs_started]() {
            g_store->StartSession();
            size_t base = (size_t)t * 50;
            for (size_t p = 0; p < 25 && base + p * 2 + 1 < kNumKeys &&
                 writers_running.load(std::memory_order_relaxed); p++) {
                uint64_t k1 = base + p * 2;
                uint64_t k2 = base + p * 2 + 1;
                g_store->RMW(k1, kModData, kValueSize);
                g_store->RMW(k2, kModData, kValueSize);
                thread_pairs[t].push_back({k1, k2});
                pairs_started.fetch_add(1, std::memory_order_relaxed);
                if ((p % kRefreshInterval) == 0) g_store->Refresh();
            }
            g_store->StopSession();
        });
    }

    // Wait until writers have made observable progress before taking the
    // checkpoint. Without this, the checkpoint can race ahead of all writes
    // and trivially recover the initial state only.
    for (int spin = 0; spin < 100 && pairs_started.load(std::memory_order_relaxed) == 0; ++spin) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // Checkpoint while writers may still be active.
    g_store->Checkpoint();

    writers_running.store(false);
    for (auto& th : threads) th.join();

    // Simulate crash + recovery
    delete g_store;
    g_store = create_kvstore();
    g_store->InitExtended(kHashTableSize, kLogSize, 0, storage_path);

    verify_monotonicity("crash_recovery_fuzzy", thread_pairs, num_threads,
                        /*require_liveness=*/true);
}

// ── TEST 2: No-checkpoint crash recovery ────────────────────────────────────
// Checkpoint initial state, do paired writes, crash WITHOUT new checkpoint.
static void test_crash_recovery_no_ckpt(int num_threads, const char* storage_path) {
    fprintf(stderr, "=== TEST 2: CRASH RECOVERY (No Checkpoint, small scale) ===\n");

    load_keys();
    g_store->Checkpoint();  // checkpoint the initial state

    std::vector<std::vector<PairRecord>> thread_pairs(num_threads);

    {
        std::vector<std::thread> threads;
        for (int t = 0; t < num_threads; t++) {
            threads.emplace_back([t, &thread_pairs]() {
                g_store->StartSession();
                size_t base = (size_t)t * 50;
                for (size_t p = 0; p < 25 && base + p * 2 + 1 < kNumKeys; p++) {
                    uint64_t k1 = base + p * 2;
                    uint64_t k2 = base + p * 2 + 1;
                    g_store->RMW(k1, kModData, kValueSize);
                    g_store->RMW(k2, kModData, kValueSize);
                    thread_pairs[t].push_back({k1, k2});
                    if ((p % kRefreshInterval) == 0) g_store->Refresh();
                }
                g_store->StopSession();
            });
        }
        for (auto& th : threads) th.join();
    }

    // NO checkpoint — simulate abrupt crash
    delete g_store;
    g_store = create_kvstore();
    g_store->InitExtended(kHashTableSize, kLogSize, 0, storage_path);

    verify_monotonicity("crash_recovery_no_ckpt", thread_pairs, num_threads,
                        /*require_liveness=*/false);
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <test_id> <num_threads> [storage_path]\n"
                "  1=crash_recovery_fuzzy  2=crash_recovery_no_ckpt\n", argv[0]);
        return 1;
    }
    int test_id = atoi(argv[1]);
    int num_threads = atoi(argv[2]);
    const char* storage_path = (argc >= 4) ? argv[3] : "/tmp/unit_consistency_store";

    g_store = create_kvstore();
    g_store->InitExtended(kHashTableSize, kLogSize, 0, storage_path);

    switch (test_id) {
    case 1: test_crash_recovery_fuzzy(num_threads, storage_path); break;
    case 2: test_crash_recovery_no_ckpt(num_threads, storage_path); break;
    default: fprintf(stderr, "Unknown test_id\n"); return 1;
    }

    if (g_store) {
        g_store->Checkpoint();
        delete g_store;
    }
    return 0;
}
