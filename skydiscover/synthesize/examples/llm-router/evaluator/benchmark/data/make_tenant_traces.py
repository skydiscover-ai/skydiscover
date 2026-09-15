#!/usr/bin/env python3
"""Build the two-tenant traces from LLMRouterBench (see task.md).

Tenant A is interactive (simpleqa, gpqa, tau2; bursty, strict TTFT); tenant B is batch (hle,
livecodebench, livemathbench; deadline SLO). Keeps prompts every fleet model has a measured score
and cost for, splits 60/20/20 by a hash of the prompt id, and writes committed manifests and cards
under evaluator/benchmark/data/ plus gitignored traces under evaluator/benchmark/.data/traces/. Quality and cost are
measured; arrivals are seeded. Deterministic, sha256s printed. The test split is written but never
read here.

    python evaluator/benchmark/data/make_tenant_traces.py
"""

import glob
import hashlib
import json
import math
import os
import random
import statistics

HERE = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)  # example root
BASE = os.path.join(HERE, "evaluator", "benchmark", ".data", "lrb", "bench-release")
OUT_M = os.path.join(HERE, "evaluator", "benchmark", "data", "manifests")
OUT_C = os.path.join(HERE, "evaluator", "benchmark", "data", "cards")
OUT_D = os.path.join(HERE, "evaluator", "benchmark", ".data", "traces")
FLAGSHIP = os.path.join(HERE, "evaluator", "benchmark", "data", "fleet_flagship.json")

TENANTS = {
    "A": {
        "label": "interactive analyst",
        "tasks": ["simpleqa", "gpqa", "tau2"],
        "model_requested": "gpt-5",
        "slo": {"ttft_ms": 2500},
        "stream": True,
    },
    "B": {
        "label": "batch reasoning",
        "tasks": ["hle", "livecodebench", "livemathbench"],
        "model_requested": "deepseek-r1-0528",
        "slo": {"latency_ms": 600000},
        "stream": False,
    },
}
SPLITS = ("train", "val", "test")
SEED = "round4-v1"  # FROZEN historical seed string: every trace hash derives from it

# Tenant A arrival shape: Poisson base ~0.5/s, 10x bursts, diurnal envelope
A_BASE_RPS = 0.5
A_BURST_FACTOR = 10.0
A_BURST_GAP_MEAN_S = 420.0  # exponential gap between burst starts
A_BURST_LEN_S = (15.0, 30.0)  # uniform burst duration
A_DIURNAL_PERIOD_S = 1200.0  # one compressed "day"
A_DIURNAL_AMP = 0.5  # envelope = 1 + amp*sin(2*pi*t/period)

# Tenant B arrival shape: heavy batch waves every ~10 min
B_WAVE_GAP_S = 600.0
B_WAVE_JITTER_S = 45.0
B_WAVE_RAMP_S = 90.0  # each wave's submissions land uniformly in this window


def split_of(pid):
    h = int(hashlib.sha256(pid.encode()).hexdigest(), 16) % 100
    return "train" if h < 60 else ("val" if h < 80 else "test")


def task_records(task, model):
    fs = sorted(glob.glob(f"{BASE}/{task}/test/{model}/*.json"))
    return json.load(open(fs[0]))["records"] if fs else []


def _int_or(v, fallback):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return fallback


def build_rows(tenant, fleet):
    """Per-prompt rows for one tenant, keyed and ordered deterministically."""
    rows = {s: [] for s in SPLITS}
    for task in TENANTS[tenant]["tasks"]:
        per_model = {m: {r["index"]: r for r in task_records(task, m)} for m in fleet}
        common = set.intersection(*[set(d) for d in per_model.values()])
        # drop any prompt where any fleet model has a None score or cost
        common = {
            i
            for i in common
            if all(per_model[m][i].get("score") is not None for m in fleet)
            and all(per_model[m][i].get("cost") is not None for m in fleet)
        }
        for idx in sorted(common):
            pid = f"{task}:{idx}"
            sample = per_model[fleet[0]][idx]
            prompt = str(sample.get("origin_query") or sample.get("prompt") or "")
            fallback_pt = max(16, len(prompt) // 4)
            pt = {m: _int_or(per_model[m][idx].get("prompt_tokens"), fallback_pt) for m in fleet}
            ct = {m: _int_or(per_model[m][idx].get("completion_tokens"), 64) for m in fleet}
            rows[split_of(pid)].append(
                {
                    "prompt_id": pid,
                    "task": task,
                    "tenant": tenant,
                    "prompt_head": prompt[:512],
                    "prompt_tokens": max(1, int(round(statistics.median(pt.values())))),
                    "completion_tokens": max(1, int(round(statistics.median(ct.values())))),
                    "score": {m: float(per_model[m][idx]["score"]) for m in fleet},
                    "cost": {m: float(per_model[m][idx]["cost"]) for m in fleet},
                    "prompt_tokens_by_model": pt,
                    "completion_tokens_by_model": ct,
                }
            )
    return rows


def arrivals_tenant_a(n, rng):
    """Inhomogeneous Poisson by thinning: base rate x diurnal envelope x burst factor."""
    horizon = n / A_BASE_RPS * 4 + 3600  # generous; thinning stops after n accepts
    bursts, t = [], 0.0
    while t < horizon:
        t += rng.expovariate(1.0 / A_BURST_GAP_MEAN_S)
        bursts.append((t, t + rng.uniform(*A_BURST_LEN_S)))

    def rate(ts):
        env = 1.0 + A_DIURNAL_AMP * math.sin(2 * math.pi * ts / A_DIURNAL_PERIOD_S)
        burst = any(s <= ts < e for s, e in bursts)
        return A_BASE_RPS * env * (A_BURST_FACTOR if burst else 1.0)

    lam_max = A_BASE_RPS * (1.0 + A_DIURNAL_AMP) * A_BURST_FACTOR
    out, ts = [], 0.0
    while len(out) < n:
        ts += rng.expovariate(lam_max)
        if rng.random() < rate(ts) / lam_max:
            out.append(ts * 1000.0)
    return out


def arrivals_tenant_b(n, rng, span_ms):
    """Batch waves every ~B_WAVE_GAP_S over the tenant-A span, submissions ramped."""
    span_s = max(span_ms / 1000.0, B_WAVE_GAP_S)
    n_waves = max(1, int(round(span_s / B_WAVE_GAP_S)))
    starts = [
        min(
            max(60.0, (k + 0.5) * span_s / n_waves + rng.uniform(-1, 1) * B_WAVE_JITTER_S),
            span_s - B_WAVE_RAMP_S,
        )
        for k in range(n_waves)
    ]
    out = []
    for i in range(n):
        w = starts[i % n_waves]
        out.append((w + rng.uniform(0, B_WAVE_RAMP_S)) * 1000.0)
    return sorted(out)


def to_request(row, t_ms, fleet):
    t = TENANTS[row["tenant"]]
    return {
        "req_id": None,  # assigned after the merge
        "t_ms": int(round(t_ms)),
        "session_id": f"{row['tenant']}-{row['task']}",
        "class": row["task"],
        "tenant": row["tenant"],
        "model_requested": t["model_requested"],
        "equiv_class": list(fleet),
        "prompt_tokens": row["prompt_tokens"],
        "prefix_id": f"{row['tenant']}-{row['task']}",
        "prefix_tokens": 0,  # the prefix-cache axis is switched off in this setting
        "expected_output_tokens": row["completion_tokens"],
        "max_tokens": max(256, 4 * row["completion_tokens"]),
        "stream": t["stream"],
        "slo": dict(t["slo"]),
        "downgrade_ok": True,
        "retry_safe": True,
        "temperature": 0.0,
        "quality": {m: round(row["score"][m], 4) for m in fleet},
        "quality_floor": 0.0,
        "features": {
            "tenant": row["tenant"],
            "task": row["task"],
            "prompt_id": row["prompt_id"],
            "difficulty_hint": min(1.0, row["prompt_tokens"] / 2000.0),
        },
    }


def write_jsonl(path, records):
    payload = "".join(json.dumps(r, sort_keys=True) + "\n" for r in records)
    with open(path, "w") as f:
        f.write(payload)
    return hashlib.sha256(payload.encode()).hexdigest()


def pctl(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * (len(xs) - 1)))] if xs else None


def workload_card(tenant, rows, traces):
    """Measured card: content statistics from TRAIN only, arrival statistics from
    train+val traces; the test split contributes nothing but its row count."""
    tr = rows["train"]
    fleet = sorted(tr[0]["score"])
    by_task = {}
    for r in tr:
        by_task.setdefault(r["task"], []).append(r)
    lines = [
        "# Workload card (measured from the frozen build; train/val only, test row count only)",
        f'tenant: "{tenant}"',
        f'label: "{TENANTS[tenant]["label"]}"',
        f"tasks: {json.dumps(TENANTS[tenant]['tasks'])}",
        f'model_requested: "{TENANTS[tenant]["model_requested"]}"',
        f"slo: {json.dumps(TENANTS[tenant]['slo'])}",
        f"downgrade_ok: true",
        f"rows: {{train: {len(rows['train'])}, val: {len(rows['val'])}, test: {len(rows['test'])}}}",
        "arrivals:",
    ]
    for split in ("train", "val"):
        ts = [r["t_ms"] / 1000.0 for r in traces[split]]
        span = max(ts) - min(ts) if ts else 0.0
        peak = 0
        if ts:
            bins = {}
            for x in ts:
                bins[int(x // 10)] = bins.get(int(x // 10), 0) + 1
            peak = max(bins.values()) / 10.0
        lines.append(
            f"  {split}: {{n: {len(ts)}, span_sec: {span:.1f}, "
            f"mean_rps: {len(ts) / span if span else 0:.3f}, peak_rps_10s: {peak:.1f}}}"
        )
    lines.append(
        "prompt_tokens_p50_p90_p99: "
        + json.dumps([pctl([r["prompt_tokens"] for r in tr], q) for q in (0.5, 0.9, 0.99)])
    )
    lines.append(
        "output_tokens_p50_p90_p99: "
        + json.dumps([pctl([r["completion_tokens"] for r in tr], q) for q in (0.5, 0.9, 0.99)])
    )
    lines.append("tasks_train_share:")
    for t in TENANTS[tenant]["tasks"]:
        lines.append(f"  {t}: {len(by_task.get(t, [])) / len(tr):.3f}")
    lines.append("# per-model measured mean score / mean cost per request, TRAIN split")
    lines.append("models_train:")
    for m in fleet:
        q = statistics.mean(r["score"][m] for r in tr)
        c = statistics.mean(r["cost"][m] for r in tr)
        lines.append(f"  {m}: {{mean_score: {q:.4f}, mean_cost_usd: {c:.6f}}}")
    return "\n".join(lines) + "\n"


def main():
    os.makedirs(OUT_M, exist_ok=True)
    os.makedirs(OUT_C, exist_ok=True)
    os.makedirs(OUT_D, exist_ok=True)
    fleet = json.load(open(FLAGSHIP))["fleet"]
    assert len(fleet) == 12, fleet

    rows = {t: build_rows(t, fleet) for t in TENANTS}
    hashes = {}

    for split in SPLITS:
        # per-tenant matrices + manifests
        for t in TENANTS:
            man = {
                "tenant": t,
                "label": TENANTS[t]["label"],
                "split": split,
                "tasks": TENANTS[t]["tasks"],
                "fleet": fleet,
                "hash_rule": "sha256(prompt_id) % 100 -> <60 train, <80 val, else test",
                "prompt_ids": [r["prompt_id"] for r in rows[t][split]],
            }
            mp = f"{OUT_M}/tenant{t}_{split}.manifest.json"
            payload = json.dumps(man, sort_keys=True)
            open(mp, "w").write(payload)
            hashes[os.path.basename(mp)] = hashlib.sha256(payload.encode()).hexdigest()
            hashes[f"matrix_tenant{t}_{split}.jsonl"] = write_jsonl(
                f"{OUT_D}/matrix_tenant{t}_{split}.jsonl", rows[t][split]
            )

        # arrivals: A first (defines the span), then B's waves fill the same span
        rng_a = random.Random(f"{SEED}-A-{split}")
        rng_b = random.Random(f"{SEED}-B-{split}")
        a_rows = list(rows["A"][split])
        b_rows = list(rows["B"][split])
        rng_a.shuffle(a_rows)
        rng_b.shuffle(b_rows)
        t_a = arrivals_tenant_a(len(a_rows), rng_a)
        t_b = arrivals_tenant_b(len(b_rows), rng_b, max(t_a) if t_a else 600000.0)

        merged = [to_request(r, ts, fleet) for r, ts in zip(a_rows, t_a)] + [
            to_request(r, ts, fleet) for r, ts in zip(b_rows, t_b)
        ]
        merged.sort(key=lambda r: (r["t_ms"], r["tenant"], r["features"]["prompt_id"]))
        for k, r in enumerate(merged):
            r["req_id"] = k
        hashes[f"trace_merged_{split}.jsonl"] = write_jsonl(
            f"{OUT_D}/trace_merged_{split}.jsonl", merged
        )
        for t in TENANTS:
            view = [r for r in merged if r["tenant"] == t]
            hashes[f"trace_tenant{t}_{split}.jsonl"] = write_jsonl(
                f"{OUT_D}/trace_tenant{t}_{split}.jsonl", view
            )
        counts = {t: sum(1 for r in merged if r["tenant"] == t) for t in TENANTS}
        print(
            f"{split:5s} merged n={len(merged)} span={merged[-1]['t_ms'] / 1000:.0f}s "
            f"A={counts['A']} B={counts['B']}"
        )

    # measured cards (train/val content only)
    for t in TENANTS:
        traces = {}
        for split in ("train", "val"):
            traces[split] = [json.loads(l) for l in open(f"{OUT_D}/trace_tenant{t}_{split}.jsonl")]
        card = workload_card(t, rows[t], traces)
        cp = f"{OUT_C}/tenant{t}.card.yaml"
        open(cp, "w").write(card)
        hashes[os.path.basename(cp)] = hashlib.sha256(card.encode()).hexdigest()

    # .data-root convenience symlink to the merged train trace.
    link = os.path.join(HERE, "evaluator", "benchmark", ".data", "trace_merged_train.jsonl")
    if not os.path.islink(link) and not os.path.exists(link):
        os.symlink(os.path.join("traces", "trace_merged_train.jsonl"), link)

    print(json.dumps(hashes, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
