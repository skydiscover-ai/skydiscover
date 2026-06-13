// unit_bench_harness.cc — Small-scale benchmark harness for unit tests.
// Same logic as benchmark_harness.cc but with 1000 keys, 2s runs, inline key generation.
// Output format is identical so analyze.py can parse it.

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

#include "../../../interface/kvstore_interface.h"

static constexpr size_t kInitCount  = 1000;
static constexpr size_t kTxnCount   = 10000;
static constexpr size_t kChunkSize  = 100;
static constexpr size_t kRefreshInterval = 64;
static constexpr int    kRunSeconds = 2;
static constexpr size_t kValueSize  = 8;
static constexpr size_t kHashTableSize = 2048;
static constexpr size_t kLogSize    = 1ULL << 20;  // 1 MB
static constexpr uint64_t kInitialValue = 42;

enum Workload { A_50_50 = 0, RMW_100 = 1, B_95_5 = 2, C_100_0 = 3,
                /* 4 = W_0_100 in benchmark_harness; unit_bench keeps YCSB focus */
                TIMESERIES_HD = 5 };

// Time-series state -- mirrors benchmark_harness.cc semantics at toy scale.
static std::atomic<uint64_t> g_ts_tail{0};
static std::atomic<uint64_t> g_ts_head{0};
static std::atomic<uint64_t> g_ts_insert_ops{0};
static std::atomic<uint64_t> g_ts_delete_hits{0};
static std::atomic<uint64_t> g_ts_delete_misses{0};
static double g_ts_delete_prob = 0.5;
static size_t g_ts_safety_margin = 0;

static IKVStore*   g_store     = nullptr;
static uint64_t*   g_load_keys = nullptr;
static uint64_t*   g_run_keys  = nullptr;
static Workload    g_workload;
static std::atomic<size_t>   g_chunk_idx{0};
static std::atomic<bool>     g_running{false};
static std::atomic<uint64_t> g_total_ops{0};
static const uint8_t kModData[8] = {5, 0, 0, 0, 0, 0, 0, 0};

static uint64_t* gen_sequential_keys(size_t count) {
    uint64_t* buf = new uint64_t[count];
    for (size_t i = 0; i < count; ++i) buf[i] = i;
    return buf;
}

static uint64_t* gen_zipf_keys(size_t count, size_t n_keys) {
    uint64_t* buf = new uint64_t[count];
    std::mt19937_64 rng(42);
    for (size_t i = 0; i < count; ++i) {
        // Simple approximation of Zipfian: geometric distribution biased toward low keys
        double u = std::uniform_real_distribution<double>(0.0, 1.0)(rng);
        buf[i] = static_cast<uint64_t>(std::pow(u, 3.0) * n_keys) % n_keys;
    }
    return buf;
}

static inline bool should_read(Workload workload, std::mt19937& rng) {
    switch (workload) {
    case A_50_50: return (rng() % 100) < 50;
    case B_95_5:  return (rng() % 100) < 95;
    case C_100_0: return true;
    case RMW_100: return false;
    case TIMESERIES_HD: return false;
    }
    return false;
}

static void load_worker(int thread_id, int num_threads) {
    g_store->StartSession();
    size_t per = kInitCount / num_threads;
    size_t start = (size_t)thread_id * per;
    size_t end = (thread_id == num_threads - 1) ? kInitCount : start + per;

    GenValue val{};
    val.size = kValueSize;
    std::memcpy(val.data, &kInitialValue, sizeof(kInitialValue));

    for (size_t i = start; i < end; ++i) {
        g_store->Upsert(g_load_keys[i], val);
        if ((i % kRefreshInterval) == 0) g_store->Refresh();
    }
    g_store->StopSession();
}

static void run_worker(int /*thread_id*/) {
    g_store->StartSession();
    uint64_t ops = 0;
    uint64_t local_ts_insert = 0, local_ts_delete_hit = 0, local_ts_delete_miss = 0;
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    GenValue val{};
    val.size = kValueSize;
    memset(val.data, 0, kValueSize);

    if (g_workload == TIMESERIES_HD) {
        while (g_running.load(std::memory_order_relaxed)) {
            for (size_t i = 0; i < kChunkSize; ++i) {
                bool want_delete = (u01(rng) < g_ts_delete_prob);
                if (want_delete) {
                    uint64_t t_snap = g_ts_tail.load(std::memory_order_relaxed);
                    uint64_t h_snap = g_ts_head.load(std::memory_order_relaxed);
                    if (h_snap + g_ts_safety_margin >= t_snap) {
                        want_delete = false;  // force insert
                    }
                }
                if (want_delete) {
                    uint64_t h = g_ts_head.fetch_add(1, std::memory_order_relaxed);
                    if (g_store->Delete(h)) ++local_ts_delete_hit;
                    else                    ++local_ts_delete_miss;
                } else {
                    uint64_t t = g_ts_tail.fetch_add(1, std::memory_order_relaxed);
                    val.size = kValueSize;
                    std::memcpy(val.data, &kInitialValue, sizeof(kInitialValue));
                    g_store->Upsert(t, val);
                    ++local_ts_insert;
                }
                ++ops;
                if ((ops % kRefreshInterval) == 0) g_store->Refresh();
            }
        }
        g_total_ops.fetch_add(ops, std::memory_order_relaxed);
        g_ts_insert_ops.fetch_add(local_ts_insert, std::memory_order_relaxed);
        g_ts_delete_hits.fetch_add(local_ts_delete_hit, std::memory_order_relaxed);
        g_ts_delete_misses.fetch_add(local_ts_delete_miss, std::memory_order_relaxed);
        g_store->StopSession();
        return;
    }

    while (g_running.load(std::memory_order_relaxed)) {
        size_t chunk = g_chunk_idx.fetch_add(1, std::memory_order_relaxed);
        size_t base = (chunk * kChunkSize) % (kTxnCount - kChunkSize);
        for (size_t i = base; i < base + kChunkSize; ++i) {
            uint64_t key = g_run_keys[i];
            switch (g_workload) {
            case RMW_100: g_store->RMW(key, kModData, kValueSize); break;
            case A_50_50:
            case B_95_5:
                if (should_read(g_workload, rng)) g_store->Read(key, val);
                else g_store->Upsert(key, val);
                break;
            case C_100_0: g_store->Read(key, val); break;
            case TIMESERIES_HD: break;  // handled above; unreachable here
            }
            ++ops;
            if ((ops % kRefreshInterval) == 0) g_store->Refresh();
        }
    }
    g_total_ops.fetch_add(ops, std::memory_order_relaxed);
    g_store->StopSession();
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <workload_id> <num_threads>\n", argv[0]);
        return 1;
    }
    int workload_id = atoi(argv[1]);
    int num_threads = atoi(argv[2]);
    g_workload = static_cast<Workload>(workload_id);

    g_load_keys = gen_sequential_keys(kInitCount);
    g_run_keys  = gen_zipf_keys(kTxnCount, kInitCount);

    g_store = create_kvstore();
    g_store->Init(kHashTableSize, kLogSize);

    // Load
    {
        std::vector<std::thread> threads;
        for (int i = 0; i < num_threads; ++i)
            threads.emplace_back(load_worker, i, num_threads);
        for (auto& t : threads) t.join();
    }

    // Validate
    fprintf(stderr, "Validating...\n");
    uint64_t checked = 0, retained = 0, missing = 0, wrong_size = 0, wrong_value = 0;
    g_store->StartSession();
    for (size_t i = 0; i < kInitCount; ++i) {
        checked++;
        GenValue out{};
        if (!g_store->Read(g_load_keys[i], out)) { missing++; continue; }
        uint64_t v = 0;
        if (out.size != 8) { wrong_size++; continue; }
        std::memcpy(&v, out.data, 8);
        if (v != kInitialValue) { wrong_value++; continue; }
        retained++;
        if ((i % kRefreshInterval) == 0) g_store->Refresh();
    }
    g_store->StopSession();

    double retention_pct = checked ? 100.0 * retained / checked : 0;
    fprintf(stderr, "VALIDATION load_keys checked=%lu retained=%lu missing=%lu "
            "wrong_size=%lu wrong_value=%lu retention_pct=%.6f elapsed_sec=0.000 stride=1\n",
            checked, retained, missing, wrong_size, wrong_value, retention_pct);

    if (missing > 0 || wrong_size > 0 || wrong_value > 0) {
        fprintf(stderr, "VALIDATION FAILED\n");
        delete g_store; delete[] g_load_keys; delete[] g_run_keys;
        return 2;
    }
    fprintf(stderr, "VALIDATION PASSED: retained %lu / %lu checked load keys.\n", retained, checked);

    // Time-series init (no-op for other workloads; cheap globals).
    if (g_workload == TIMESERIES_HD) {
        g_ts_tail.store(kInitCount);
        g_ts_head.store(0);
        g_ts_insert_ops.store(0);
        g_ts_delete_hits.store(0);
        g_ts_delete_misses.store(0);
        g_ts_safety_margin = std::max<size_t>(
            static_cast<size_t>(num_threads) * 4u, kInitCount / 4);
        const char* raw = std::getenv("KVSTORE_DELETE_RATE");
        double r = 1.0;
        if (raw && *raw) {
            char* end = nullptr;
            double parsed = std::strtod(raw, &end);
            if (end && *end == '\0' && parsed >= 0.0) r = parsed;
        }
        g_ts_delete_prob = r / (1.0 + r);
    }

    // Run
    g_chunk_idx.store(0);
    g_total_ops.store(0);
    g_running.store(true);
    auto t0 = std::chrono::steady_clock::now();
    {
        std::vector<std::thread> threads;
        for (int i = 0; i < num_threads; ++i)
            threads.emplace_back(run_worker, i);
        std::this_thread::sleep_for(std::chrono::seconds(kRunSeconds));
        g_running.store(false);
        for (auto& t : threads) t.join();
    }
    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    double total_ops = static_cast<double>(g_total_ops.load());
    double ops_per_thread = total_ops / secs / num_threads;

    printf("Finished benchmark: 0 thread checkpoints completed;  %.2f ops/second/thread\n", ops_per_thread);

    // Time-series integrity: sample live / deleted keys, fail if any mismatch.
    int exit_code = 0;
    if (g_workload == TIMESERIES_HD) {
        uint64_t ins  = g_ts_insert_ops.load();
        uint64_t dh   = g_ts_delete_hits.load();
        uint64_t dm   = g_ts_delete_misses.load();
        uint64_t head = g_ts_head.load();
        uint64_t tail = g_ts_tail.load();
        fprintf(stderr,
                "TS stats: inserts=%lu delete_hits=%lu delete_misses=%lu "
                "head=%lu tail=%lu live=%lu\n",
                ins, dh, dm, head, tail, tail - head);
        // Require at least 1 successful insert AND 1 successful delete
        // so the test binary actually exercises both code paths.
        if (ins == 0 || dh == 0) {
            fprintf(stderr,
                    "TS INTEGRITY FAILED: need ins>0 and delete_hits>0 (got %lu/%lu)\n",
                    ins, dh);
            exit_code = 3;  // distinct from benchmark_harness.cc's exit 2
        }
        // Live/deleted spot-check at small scale
        g_store->StartSession();
        std::mt19937_64 rng{0xD1CE5EEDULL};
        GenValue out{};
        size_t sampled_live = 0, live_missing = 0;
        size_t sampled_deleted = 0, deleted_present = 0;
        size_t live_range = (tail > head) ? static_cast<size_t>(tail - head) : 0;
        size_t deleted_range = static_cast<size_t>(head);
        size_t sample_n = 256;
        for (size_t i = 0; i < std::min(sample_n, live_range); ++i) {
            uint64_t k = head + (rng() % live_range);
            ++sampled_live;
            if (!g_store->Read(k, out)) ++live_missing;
        }
        for (size_t i = 0; i < std::min(sample_n, deleted_range); ++i) {
            uint64_t k = rng() % deleted_range;
            ++sampled_deleted;
            if (g_store->Read(k, out)) ++deleted_present;
        }
        g_store->StopSession();
        fprintf(stderr,
                "TS live-set: live_sampled=%zu missing=%zu  deleted_sampled=%zu readable=%zu\n",
                sampled_live, live_missing, sampled_deleted, deleted_present);
        if (live_missing > 0 || deleted_present > 0) {
            fprintf(stderr, "TS INTEGRITY FAILED: live-set violations "
                    "(live_missing=%zu deleted_present=%zu)\n",
                    live_missing, deleted_present);
            exit_code = 3;
        }
    }

    delete g_store; delete[] g_load_keys; delete[] g_run_keys;
    return exit_code;
}
