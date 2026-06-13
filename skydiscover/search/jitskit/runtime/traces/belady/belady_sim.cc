// belady_sim.cc — Belady's optimal (MIN) cache simulation.
//
// Reads the same binary trace files as kvstore_bench, simulates Belady
// optimal eviction at a given cache capacity, and prints cache hit metrics
// in the same format as the benchmark harness.
//
// Belady's algorithm: on a cache miss, evict the item whose next access
// is furthest in the future. This is the theoretical upper bound — no
// real eviction policy can achieve a higher hit ratio for a given trace
// and cache size.
//
// Usage:
//   ./belady_sim <workload> <cache_items> <load_file> <run_file> [max_ops]
//
//   workload:     "5050" | "rmw" | "read_only" | "upsert_only"
//   cache_items:  number of key-value pairs that fit in memory
//                 (e.g., for 8GB budget with 3GB hash overhead and 100B values:
//                  (8G - 3G) / 100 = 50000000)
//   load_file:    binary file of uint64_t keys (250M)
//   run_file:     binary file of uint64_t keys (1B)
//   max_ops:      optional cap on run-phase ops (default: all)
//
// Output (matches benchmark_harness.cc format):
//   ReadCacheHit:    XX.XX% (hits / total)
//   RmwCacheHit:     YY.YY% (hits / total)
//   CacheHitTotal:   ZZ.ZZ% (hits / total)   [only when both exist]
//   CacheSizeRatio:  A.AA / B.BB GB (PP.P%)
//   CacheBudgetUtil: [not printed — use cache_items × value_size for budget]
//
// Build:
//   g++ -O3 -std=c++17 -o belady_sim belady_sim.cc

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

// ── Load binary trace ────────────────────────────────────────────────────────
static std::vector<uint64_t> load_trace(const char* path, size_t max_count = 0) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "ERROR: cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END);
    size_t file_bytes = ftell(f);
    fseek(f, 0, SEEK_SET);
    size_t file_count = file_bytes / sizeof(uint64_t);
    size_t count = (max_count > 0 && max_count < file_count) ? max_count : file_count;
    std::vector<uint64_t> keys(count);
    size_t got = fread(keys.data(), sizeof(uint64_t), count, f);
    fclose(f);
    if (got != count) {
        fprintf(stderr, "WARN: %s: expected %zu keys, got %zu\n", path, count, got);
        keys.resize(got);
    }
    return keys;
}

// ── Op types ─────────────────────────────────────────────────────────────────
enum OpType { OP_READ, OP_UPSERT, OP_RMW };

// Assign op types based on workload string
static std::vector<OpType> assign_ops(const std::string& workload, size_t n) {
    std::vector<OpType> ops(n);
    if (workload == "rmw") {
        std::fill(ops.begin(), ops.end(), OP_RMW);
    } else if (workload == "read_only") {
        std::fill(ops.begin(), ops.end(), OP_READ);
    } else if (workload == "upsert_only") {
        std::fill(ops.begin(), ops.end(), OP_UPSERT);
    } else {
        // 5050: 50% read, 50% upsert (deterministic alternation for reproducibility)
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> coin(0, 1);
        for (size_t i = 0; i < n; i++) {
            ops[i] = coin(rng) ? OP_READ : OP_UPSERT;
        }
    }
    return ops;
}

int main(int argc, char* argv[]) {
    if (argc < 5) {
        fprintf(stderr,
            "Usage: %s <workload> <cache_items> <load_file> <run_file> [max_ops]\n"
            "  workload:    5050 | rmw | read_only | upsert_only\n"
            "  cache_items: cache capacity in number of key-value pairs\n"
            "  load_file:   binary uint64_t trace (250M keys)\n"
            "  run_file:    binary uint64_t trace (1B keys)\n"
            "  max_ops:     optional cap on run-phase ops (default: all)\n"
            "\n"
            "Example (8GB budget, 3GB hash overhead, 100B values):\n"
            "  %s 5050 50000000 load_zipf_250M_raw.dat run_zipf_250M_1000M_raw.dat\n",
            argv[0], argv[0]);
        return 1;
    }

    std::string workload = argv[1];
    size_t cache_cap = std::strtoull(argv[2], nullptr, 10);
    const char* load_path = argv[3];
    const char* run_path  = argv[4];
    size_t max_ops = (argc > 5) ? std::strtoull(argv[5], nullptr, 10) : 0;

    fprintf(stderr, "Belady simulation: workload=%s, cache=%zu items\n",
            workload.c_str(), cache_cap);

    // ── Load traces ──────────────────────────────────────────────────────────
    auto t0 = std::chrono::steady_clock::now();

    fprintf(stderr, "Loading load trace from %s ...\n", load_path);
    auto load_keys = load_trace(load_path);
    fprintf(stderr, "  %zu load keys\n", load_keys.size());

    fprintf(stderr, "Loading run trace from %s ...\n", run_path);
    auto run_keys = load_trace(run_path, max_ops);
    fprintf(stderr, "  %zu run keys\n", run_keys.size());

    size_t N = run_keys.size();
    auto ops = assign_ops(workload, N);

    auto t1 = std::chrono::steady_clock::now();
    double load_secs = std::chrono::duration<double>(t1 - t0).count();
    fprintf(stderr, "Traces loaded in %.1f s\n", load_secs);

    // ── Pre-compute next_access for run trace ────────────────────────────────
    // For each position i in the run trace, next_access[i] = the next position
    // j > i where run_keys[j] == run_keys[i], or INT64_MAX if no such j.
    fprintf(stderr, "Computing next-access array (%zu ops) ...\n", N);
    std::vector<int64_t> next_access(N, INT64_MAX);
    {
        std::unordered_map<uint64_t, int64_t> last_seen;
        last_seen.reserve(300'000'000);
        // Scan backwards
        for (int64_t i = static_cast<int64_t>(N) - 1; i >= 0; i--) {
            auto it = last_seen.find(run_keys[i]);
            if (it != last_seen.end()) {
                next_access[i] = it->second;
            }
            last_seen[run_keys[i]] = i;
        }
        // Also compute next_access for load keys that appear in run trace
        // (so initial cache eviction during load is Belady-optimal w.r.t. run phase)
    }

    auto t2 = std::chrono::steady_clock::now();
    fprintf(stderr, "Next-access computed in %.1f s\n",
            std::chrono::duration<double>(t2 - t1).count());

    // ── Build initial next-access map for load keys ──────────────────────────
    // For keys in the load trace, their "next access" is their first appearance
    // in the run trace (if any), or INT64_MAX.
    std::unordered_map<uint64_t, int64_t> load_next;
    load_next.reserve(300'000'000);
    {
        // First appearance in run trace
        std::unordered_map<uint64_t, int64_t> first_in_run;
        first_in_run.reserve(300'000'000);
        for (int64_t i = static_cast<int64_t>(N) - 1; i >= 0; i--) {
            first_in_run[run_keys[i]] = i;  // last write wins = earliest position
        }
        for (size_t i = 0; i < load_keys.size(); i++) {
            auto it = first_in_run.find(load_keys[i]);
            load_next[load_keys[i]] = (it != first_in_run.end()) ? it->second : INT64_MAX;
        }
    }

    auto t3 = std::chrono::steady_clock::now();
    fprintf(stderr, "Load next-access map built in %.1f s\n",
            std::chrono::duration<double>(t3 - t2).count());

    // ── Simulate load phase with Belady ──────────────────────────────────────
    // Insert all load keys into cache. When cache is full, evict the key whose
    // first run-phase access is farthest away.
    fprintf(stderr, "Simulating load phase (%zu keys, cache=%zu) ...\n",
            load_keys.size(), cache_cap);

    // Cache: set of (next_access_in_run, key). Max element = eviction candidate.
    std::set<std::pair<int64_t, uint64_t>> cache;
    std::unordered_map<uint64_t, int64_t> in_cache;  // key → its next_access in the set
    in_cache.reserve(cache_cap + 1);

    for (size_t i = 0; i < load_keys.size(); i++) {
        uint64_t key = load_keys[i];
        int64_t na = load_next.count(key) ? load_next[key] : INT64_MAX;

        // If already in cache, update its next_access
        auto it = in_cache.find(key);
        if (it != in_cache.end()) {
            cache.erase({it->second, key});
            cache.insert({na, key});
            it->second = na;
            continue;
        }

        // Insert
        if (cache.size() < cache_cap) {
            cache.insert({na, key});
            in_cache[key] = na;
        } else {
            // Evict the key with farthest next access
            auto victim = cache.end();
            --victim;
            uint64_t vkey = victim->second;
            cache.erase(victim);
            in_cache.erase(vkey);
            // Insert new
            cache.insert({na, key});
            in_cache[key] = na;
        }

        if ((i & 0xFFFFFF) == 0 && i > 0) {
            fprintf(stderr, "  load: %zuM / %zuM\r", i / 1'000'000, load_keys.size() / 1'000'000);
        }
    }

    auto t4 = std::chrono::steady_clock::now();
    fprintf(stderr, "Load phase done in %.1f s. Cache: %zu items\n",
            std::chrono::duration<double>(t4 - t3).count(), cache.size());

    // Free load-phase structures
    size_t load_count = load_keys.size();
    load_keys.clear(); load_keys.shrink_to_fit();
    load_next.clear();

    // ── Simulate run phase with Belady ───────────────────────────────────────
    fprintf(stderr, "Simulating run phase (%zu ops, cache=%zu) ...\n", N, cache.size());

    uint64_t read_hits = 0, read_misses = 0;
    uint64_t rmw_hits = 0, rmw_misses = 0;

    for (size_t i = 0; i < N; i++) {
        uint64_t key = run_keys[i];
        OpType op = ops[i];
        int64_t na = next_access[i];  // next access of THIS key after position i

        bool in = in_cache.count(key) > 0;

        if (op == OP_READ) {
            if (in) read_hits++;
            else    read_misses++;
        } else if (op == OP_RMW) {
            if (in) rmw_hits++;
            else    rmw_misses++;
        }
        // OP_UPSERT: blind write, doesn't contribute to hit ratio

        // Update cache state: key is now accessed, update its next_access
        if (in) {
            int64_t old_na = in_cache[key];
            cache.erase({old_na, key});
            cache.insert({na, key});
            in_cache[key] = na;
        } else {
            // Cache miss (or upsert of new key) — bring into cache
            if (cache.size() >= cache_cap) {
                auto victim = cache.end(); --victim;
                in_cache.erase(victim->second);
                cache.erase(victim);
            }
            cache.insert({na, key});
            in_cache[key] = na;
        }

        if ((i & 0xFFFFFF) == 0 && i > 0) {
            fprintf(stderr, "  run: %zuM / %zuM\r", i / 1'000'000, N / 1'000'000);
        }
    }

    auto t5 = std::chrono::steady_clock::now();
    double run_secs = std::chrono::duration<double>(t5 - t4).count();
    double total_secs = std::chrono::duration<double>(t5 - t0).count();
    fprintf(stderr, "Run phase done in %.1f s (total %.1f s)\n", run_secs, total_secs);

    // ── Output (same format as benchmark_harness.cc) ─────────────────────────
    printf("Belady optimal simulation: workload=%s, cache=%zu items\n",
           workload.c_str(), cache_cap);

    uint64_t total_reads = read_hits + read_misses;
    uint64_t total_rmws  = rmw_hits + rmw_misses;

    if (total_reads > 0) {
        printf("ReadCacheHit:\t%.2f%% (%llu / %llu)\n",
               100.0 * read_hits / total_reads,
               (unsigned long long)read_hits, (unsigned long long)total_reads);
    }
    if (total_rmws > 0) {
        printf("RmwCacheHit:\t%.2f%% (%llu / %llu)\n",
               100.0 * rmw_hits / total_rmws,
               (unsigned long long)rmw_hits, (unsigned long long)total_rmws);
    }
    if (total_reads > 0 && total_rmws > 0) {
        uint64_t all_hits = read_hits + rmw_hits;
        uint64_t all_ops  = total_reads + total_rmws;
        printf("CacheHitTotal:\t%.2f%% (%llu / %llu)\n",
               100.0 * all_hits / all_ops,
               (unsigned long long)all_hits, (unsigned long long)all_ops);
    }

    // Summary
    uint64_t all_hits = read_hits + rmw_hits;
    uint64_t all_misses = read_misses + rmw_misses;
    uint64_t all_ops = all_hits + all_misses;
    if (all_ops > 0) {
        printf("Summary:\t%llu hits, %llu misses, %.2f%% overall hit rate\n",
               (unsigned long long)all_hits, (unsigned long long)all_misses,
               100.0 * all_hits / all_ops);
    }
    printf("CacheCapacity:\t%zu items\n", cache_cap);
    printf("TraceOps:\t%zu (load=%zu, run=%zu)\n",
           N + load_count, load_count, N);
    printf("SimTime:\t%.1f s\n", total_secs);

    return 0;
}
