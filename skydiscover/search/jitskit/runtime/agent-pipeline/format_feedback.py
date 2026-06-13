#!/usr/bin/env python3
"""
agent-pipeline/format_feedback.py -- Format benchmark results into agent feedback.

Parses raw benchmark output files and produces a text summary the agent
reads between iterations. Replaces the inline shell parsing in run.sh.

Usage:
  python3 analysis/format_feedback.py \
    --bench-dir <path>          # directory with wlN_tM[_memXg].txt files
    --workload-id <int>         # 0=50:50, 1=RMW, 3=100:0, 4=0:100
    --setup-name <str>          # e.g. "50:50"
    --threads <str>             # space-separated thread counts, e.g. "16"
    --mem-budgets <str>         # space-separated GB, e.g. "8 32" or "0"
    --baseline-csv <path>       # fig10_baseline_results.csv
    --baseline-name <str>       # e.g. "FASTER"
    --baseline-col <int>        # column index in CSV for this workload
"""

import argparse
import csv
import os
import re
import sys


def parse_result_file(path: str) -> dict:
    """Parse a single benchmark output file into a dict of metrics."""
    result = {}
    if not os.path.isfile(path):
        return result

    text = open(path).read()

    # Throughput
    m = re.search(r'([\d.]+)\s+Mops/s', text)
    if m:
        result['mops'] = m.group(1)

    m = re.search(r'([\d.]+)\s+ops/second/thread', text)
    if m:
        result['ops_per_thread'] = m.group(1)

    # Load phase
    m = re.search(r'Load done in ([\d.]+)', text)
    if m:
        result['load_time'] = m.group(1)

    m = re.search(r'([\d.]+)\s+M keys/s', text)
    if m:
        result['load_rate'] = m.group(1)

    # Validation
    m = re.search(r'elapsed_sec=([\d.]+)', text)
    if m:
        result['validate_time'] = m.group(1)

    # Store memory
    m = re.search(r'StoreMemUtil:\s*(.*)', text)
    if m:
        result['store_mem'] = m.group(1).strip()

    # Per-operation breakdown
    m = re.search(r'OpBreakdown:\s*(.*)', text)
    if m:
        result['op_breakdown'] = m.group(1).strip()

    # Cache statistics (populated by stores that override GetCacheStats())
    m = re.search(r'ReadCacheHit:\s*(.*)', text)
    if m:
        result['read_cache_hit'] = m.group(1).strip()
    m = re.search(r'RmwCacheHit:\s*(.*)', text)
    if m:
        result['rmw_cache_hit'] = m.group(1).strip()
    m = re.search(r'CacheHitTotal:\s*(.*)', text)
    if m:
        result['cache_hit_total'] = m.group(1).strip()
    m = re.search(r'CacheSizeRatio:\s*(.*)', text)
    if m:
        result['cache_size_ratio'] = m.group(1).strip()
    m = re.search(r'CacheBudgetUtil:\s*(.*)', text)
    if m:
        result['cache_budget_util'] = m.group(1).strip()

    # Disk I/O stats
    m = re.search(r'DiskIO:\s*(.*)', text)
    if m:
        result['disk_io'] = m.group(1).strip()

    # Per-op latency percentiles
    m = re.search(r'OpLatency:\s*(.*)', text)
    if m:
        result['op_latency'] = m.group(1).strip()

    # Eviction rate
    m = re.search(r'EvictionRate:\s*(.*)', text)
    if m:
        result['eviction_rate'] = m.group(1).strip()

    # Validation errors
    val_line = ''
    for line in text.splitlines():
        if 'VALIDATION load_keys' in line:
            val_line = line
    if val_line:
        for key in ('wrong_size', 'missing', 'wrong_value', 'retained'):
            m = re.search(rf'{key}=(\d+)', val_line)
            if m:
                result[key] = m.group(1)

    # Error detection
    if re.search(r'BENCH_TIMED_OUT|Timeout|timed out', text):
        result['error'] = 'TIMED_OUT'
        # Find last progress line
        progress_lines = re.findall(r'(Loading|Load done|Validating|VALIDATION|Running).*', text)
        if progress_lines:
            result['last_progress'] = progress_lines[-1].strip()
    elif re.search(r'Killed|SIGKILL|Cannot allocate|oom', text, re.IGNORECASE):
        result['error'] = 'OOM'
    elif re.search(r'VALIDATION FAILED', text):
        result['error'] = 'VALIDATION_FAILED'
        # Capture INTEGRITY FAILED details so the agent knows WHY
        integrity_lines = re.findall(r'INTEGRITY FAILED:.*', text)
        if integrity_lines:
            result['integrity_detail'] = '; '.join(
                l.replace('INTEGRITY FAILED: ', '').strip()
                for l in integrity_lines
            )
    elif 'mops' not in result:
        # No throughput and no recognized error — diagnose by phase.
        # Process was likely killed by cgroup OOM (SIGKILL) before printing
        # results, but the kill signal doesn't appear in redirected stdout.
        has_load_start = bool(re.search(r'Loading \d+ keys with', text))
        has_load_done = 'Load done in' in text
        has_run_start = bool(re.search(r'Running workload', text))
        if has_load_start and not has_load_done:
            result['error'] = 'KILLED_DURING_LOAD'
            result['load_phase_detail'] = (
                'killed during load phase — data structures likely exceed '
                'the memory budget during initialization')
        elif has_load_done and not has_run_start:
            result['error'] = 'KILLED_DURING_VALIDATION'
        elif has_run_start:
            result['error'] = 'KILLED_DURING_RUN'

    return result


def format_memory_feedback(store_mem_str: str, budget_gb: str) -> str:
    """Produce memory-budget utilization from the StoreMemUtil line.

    Reports raw numbers only — no prescriptions.
    """
    if not store_mem_str or budget_gb == '0':
        return ''

    m = re.match(r'([\d.]+)\s*/\s*([\d.]+)\s*GB\s*\((\d+)%', store_mem_str)
    if not m:
        return ''

    used_gb = float(m.group(1))
    total_gb = float(m.group(2))
    pct = int(m.group(3))
    unused_gb = total_gb - used_gb

    if pct <= 85:
        return (f'\n  memory_util: {used_gb:.1f} / {total_gb:.0f} GB ({pct}%), '
                f'{unused_gb:.1f} GB unused')
    elif pct > 105:
        return (f'\n  memory_util: {used_gb:.1f} / {total_gb:.0f} GB ({pct}%), '
                f'over budget by {used_gb - total_gb:.1f} GB')

    return ''  # 85-105%: nothing notable to report


def load_baselines(csv_path: str, col: int) -> dict:
    """Load baseline Mops/s per memory budget from CSV."""
    baselines = {}
    if not os.path.isfile(csv_path):
        return baselines
    with open(csv_path) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if len(row) > col:
                gb = row[0].strip()
                val = row[col].strip()
                if val and val != '0':
                    baselines[gb] = val
    return baselines


_VALID_FEEDBACK_LEVELS = ("minimal", "rich")


def parse_perf_stat_file(path: str) -> dict:
    """Parse a perf stat output file into a dict of counter → value strings.

    perf stat -x ';' writes CSV lines: value;unit;event;...
    Falls back to human-readable format if -x wasn't used.
    """
    result = {}
    if not os.path.isfile(path):
        return result
    with open(path) as f:
        text = f.read()
    for line in text.splitlines():
        # CSV format from perf stat -x ';'
        parts = line.split(';')
        if len(parts) >= 3:
            val, _unit, event = parts[0].strip(), parts[1].strip(), parts[2].strip()
            if val and event and val != '<not counted>' and val != '<not supported>':
                result[event] = val
            continue
        # Human-readable fallback
        m = re.match(r'\s*([\d,]+)\s+([\w-]+)', line)
        if m:
            val = m.group(1).replace(',', '')
            event = m.group(2)
            result[event] = val
    return result


def format_perf_feedback(perf_path: str) -> str:
    """Format perf stat output into agent-readable feedback."""
    counters = parse_perf_stat_file(perf_path)
    if not counters:
        return ''
    parts = []
    # Context switches — high = lock contention / scheduler thrashing
    if 'context-switches' in counters:
        parts.append(f'context_switches={counters["context-switches"]}')
    # Page faults — major faults = SSD I/O
    if 'page-faults' in counters:
        parts.append(f'page_faults={counters["page-faults"]}')
    if 'major-faults' in counters:
        parts.append(f'major_faults={counters["major-faults"]}')
    # CPU migrations — high = poor NUMA affinity
    if 'cpu-migrations' in counters:
        parts.append(f'cpu_migrations={counters["cpu-migrations"]}')
    # Hardware counters (available on bare-metal / vPMU-enabled VMs)
    if 'instructions' in counters and 'cycles' in counters:
        try:
            ipc = int(counters['instructions']) / max(1, int(counters['cycles']))
            parts.append(f'IPC={ipc:.2f}')
        except (ValueError, ZeroDivisionError):
            pass
    if 'cache-misses' in counters and 'cache-references' in counters:
        try:
            miss_pct = int(counters['cache-misses']) / max(1, int(counters['cache-references'])) * 100
            parts.append(f'cache_miss_rate={miss_pct:.1f}%')
        except (ValueError, ZeroDivisionError):
            pass
    if 'LLC-load-misses' in counters:
        parts.append(f'LLC_load_misses={counters["LLC-load-misses"]}')
    if 'branch-misses' in counters and 'branches' in counters:
        try:
            miss_pct = int(counters['branch-misses']) / max(1, int(counters['branches'])) * 100
            parts.append(f'branch_miss_rate={miss_pct:.1f}%')
        except (ValueError, ZeroDivisionError):
            pass
    if not parts:
        return ''
    return '    perf: ' + ', '.join(parts)


def format_feedback(bench_dir: str, workload_id: int, setup_name: str,
                    threads: list, mem_budgets: list, baselines: dict,
                    baseline_name: str,
                    level: str = "rich") -> str:
    """Format all benchmark results into agent feedback text.

    level controls how much detail the agent sees:
      "minimal" -- Mops/s + pass/fail + error type only (ablation baseline)
      "rich"    -- everything: op breakdown, cache hits, memory util, budget
                   guidance, load speed, perf stat counters (default)
    """
    lines = []
    lines.append(f'## Benchmark Results ({setup_name})')

    # Baseline reference (only when explicitly provided)
    if level == 'rich':
        for gb in mem_budgets:
            if gb in baselines:
                lines.append(f'  {baseline_name} baseline: mem={gb}GB → {baselines[gb]} Mops/s')

    for gb in mem_budgets:
        if gb != '0':
            lines.append(f'  [mem={gb}GB]')

        for t in threads:
            # Find result file
            if gb != '0':
                fname = f'wl{workload_id}_t{t}_mem{gb}g.txt'
            else:
                fname = f'wl{workload_id}_t{t}.txt'
            path = os.path.join(bench_dir, fname)

            if not os.path.isfile(path):
                # Try without mem suffix
                alt = os.path.join(bench_dir, f'wl{workload_id}_t{t}.txt')
                if os.path.isfile(alt):
                    path = alt
                else:
                    lines.append(f'  - t={t}: no output')
                    continue

            r = parse_result_file(path)

            if r.get('error') == 'TIMED_OUT':
                msg = f'  - t={t}: TIMED OUT after 900s'
                if level == 'rich':
                    if 'load_time' in r:
                        msg += f' (load={r["load_time"]}s @ {r.get("load_rate", "?")}M keys/s)'
                    elif 'last_progress' in r:
                        msg += f' (stalled at: {r["last_progress"]})'
                lines.append(msg)

            elif r.get('error') == 'OOM':
                msg = f'  - t={t}: OOM KILLED'
                lines.append(msg)
                if level == 'rich':
                    if 'store_mem' in r:
                        lines.append(f'    store_mem={r["store_mem"]}')
                    if gb != '0':
                        lines.append(f'    budget={gb}GB, cgroup_limit={int(gb)+12}GB')

            elif r.get('error') == 'VALIDATION_FAILED':
                msg = f'  - t={t}: VALIDATION FAILED'
                if level == 'rich':
                    msg += (
                        f' (wrong_size={r.get("wrong_size", "0")} '
                        f'missing={r.get("missing", "0")} '
                        f'wrong_value={r.get("wrong_value", "0")} '
                        f'retained={r.get("retained", "0")})'
                    )
                    if 'integrity_detail' in r:
                        lines.append(msg)
                        lines.append(f'    INTEGRITY: {r["integrity_detail"]}')
                    else:
                        lines.append(msg)
                else:
                    lines.append(msg)

            elif r.get('error') == 'KILLED_DURING_LOAD':
                lines.append(f'  - t={t}: KILLED DURING LOAD PHASE')
                if level == 'rich':
                    if 'load_time' in r:
                        lines.append(f'    load={r["load_time"]}s @ {r.get("load_rate", "?")}M keys/s')
                    if 'store_mem' in r:
                        lines.append(f'    store_mem={r["store_mem"]}')

            elif r.get('error') == 'KILLED_DURING_VALIDATION':
                lines.append(f'  - t={t}: KILLED DURING VALIDATION')

            elif r.get('error') == 'KILLED_DURING_RUN':
                lines.append(f'  - t={t}: KILLED DURING RUN PHASE')

            elif 'mops' in r:
                if level == 'minimal':
                    # Minimal: just Mops/s, nothing else
                    lines.append(f'  - t={t}: {r["mops"]} Mops/s')
                else:
                    # Rich: full leading indicators
                    parts = [f'{r.get("ops_per_thread", "?")} ops/s/thread']
                    if 'load_time' in r:
                        parts.append(f'load={r["load_time"]}s @ {r.get("load_rate", "?")}M keys/s')
                    if 'validate_time' in r:
                        parts.append(f'validate={r["validate_time"]}s')
                    if 'store_mem' in r:
                        parts.append(f'store_mem={r["store_mem"]}')

                    # % of baseline (only when baselines are provided)
                    ref = baselines.get(gb)
                    if ref and float(r['mops']) > 0:
                        pct = float(r['mops']) / float(ref) * 100
                        parts.append(f'{pct:.1f}% of {baseline_name}')

                    lines.append(f'  - t={t}: {r["mops"]} Mops/s ({", ".join(parts)})')

                    if 'op_breakdown' in r:
                        lines.append(f'    ops: {r["op_breakdown"]}')

                    # Cache statistics (only shown when store reports them)
                    if 'read_cache_hit' in r:
                        lines.append(f'    read_cache_hit: {r["read_cache_hit"]}')
                    if 'rmw_cache_hit' in r:
                        lines.append(f'    rmw_cache_hit: {r["rmw_cache_hit"]}')
                    if 'cache_hit_total' in r:
                        lines.append(f'    cache_hit_total: {r["cache_hit_total"]}')
                    if 'cache_size_ratio' in r:
                        lines.append(f'    cache_size_ratio: {r["cache_size_ratio"]}')
                    if 'cache_budget_util' in r:
                        lines.append(f'    cache_budget_util: {r["cache_budget_util"]}')

                    # Disk I/O stats
                    if 'disk_io' in r:
                        lines.append(f'    disk_io: {r["disk_io"]}')

                    # Per-op latency percentiles
                    if 'op_latency' in r:
                        lines.append(f'    op_latency: {r["op_latency"]}')

                    # Eviction rate
                    if 'eviction_rate' in r:
                        lines.append(f'    eviction_rate: {r["eviction_rate"]}')

                    # Memory budget utilization
                    if 'store_mem' in r:
                        mem_fb = format_memory_feedback(r['store_mem'], gb)
                        if mem_fb:
                            lines.append(mem_fb)

                    # perf stat counters (from perf_wlN_tM[_memXg].txt sidecar)
                    perf_fname = 'perf_' + os.path.basename(path)
                    perf_path = os.path.join(bench_dir, perf_fname)
                    perf_fb = format_perf_feedback(perf_path)
                    if perf_fb:
                        lines.append(perf_fb)
            else:
                lines.append(f'  - t={t}: FAILED')

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Format benchmark feedback for agent')
    parser.add_argument('--bench-dir', required=True)
    parser.add_argument('--workload-id', type=int, required=True)
    parser.add_argument('--setup-name', required=True)
    parser.add_argument('--threads', required=True, help='space-separated')
    parser.add_argument('--mem-budgets', required=True, help='space-separated GB')
    parser.add_argument('--baseline-csv', required=True)
    parser.add_argument('--baseline-name', default='FASTER')
    parser.add_argument('--baseline-col', type=int, default=1)
    parser.add_argument('--level', default='rich', choices=('minimal', 'rich'),
                        help='Feedback level: minimal (Mops/s only) or rich (all indicators)')
    args = parser.parse_args()

    threads = args.threads.split()
    mem_budgets = args.mem_budgets.split()
    baselines = load_baselines(args.baseline_csv, args.baseline_col)

    feedback = format_feedback(
        args.bench_dir, args.workload_id, args.setup_name,
        threads, mem_budgets, baselines, args.baseline_name,
        level=args.level,
    )
    print(feedback)


if __name__ == '__main__':
    main()
