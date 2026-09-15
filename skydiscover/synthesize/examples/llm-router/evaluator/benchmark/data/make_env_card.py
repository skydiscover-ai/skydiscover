#!/usr/bin/env python3
"""Generate evaluator/benchmark/env_card.yaml, the fleet the routers run against.

Per-model prices are effective per-1M-token rates fitted by non-negative least squares to the
measured cost on the train matrices, so the simulated bill tracks measured dollars. Provider tiers,
rate limits, latency, and outages are modeled; the premium tier is sized so tenant A's bursts
saturate it (10-20% of its requests see a 429). Deterministic.

    python evaluator/benchmark/data/make_env_card.py
"""

import hashlib
import json
import os
import sys

HERE = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)  # example root
sys.path.insert(0, HERE)

DATA = os.path.join(HERE, "evaluator", "benchmark", ".data", "traces")
OUT_CARD = os.path.join(HERE, "evaluator", "benchmark", "env_card.yaml")
VAL_TRACE = os.path.join(DATA, "trace_merged_val.jsonl")

FLEET = json.load(
    open(os.path.join(HERE, "evaluator", "benchmark", "data", "fleet_flagship.json"))
)["fleet"]
OPEN_WEIGHT = [
    "deepseek-r1-0528",
    "deepseek-v3-0324",
    "deepseek-v3.1-terminus",
    "glm-4.6",
    "intern-s1",
    "kimi-k2-0905",
    "qwen3-235b-a22b-2507",
    "qwen3-235b-a22b-thinking-2507",
]
# Reasoning-class models decode slower everywhere (long chains of thought): they take the
# per-model decode_tps override instead of the provider default.
REASONING = ["deepseek-r1-0528", "gemini-2.5-pro", "gpt-5", "qwen3-235b-a22b-thinking-2507"]
COURIER_DISCOUNT = 0.80  # modeled reseller margin applied to the derived base rates

# Topology constants (modeled; premium rpm/concurrency sized by the measurement below).
TOPO = {
    "value": dict(
        rpm=420,
        tpm=12_000_000,
        concurrency=96,
        ttft_base_ms=700,
        prefill_tps=25_000,
        decode_tps=70,
        reasoning_decode_tps=40,
        cache_speedup=8,
        cache_entries=500,
        error_rate=0.008,
    ),
    "prime": dict(
        rpm=132,
        tpm=3_500_000,
        concurrency=32,
        ttft_base_ms=400,
        prefill_tps=60_000,
        decode_tps=100,
        reasoning_decode_tps=55,
        cache_speedup=8,
        cache_entries=500,
        error_rate=0.002,
    ),
    "courier": dict(
        rpm=150,
        tpm=5_000_000,
        concurrency=48,
        ttft_base_ms=1700,
        prefill_tps=12_000,
        decode_tps=40,
        reasoning_decode_tps=25,
        cache_speedup=8,
        cache_entries=500,
        error_rate=0.02,
    ),
}
VAL_OUTAGES = [[300_000, 420_000], [780_000, 930_000]]  # premium down: 120 s + 150 s
TEST_SEED = 20260820
RETRY_GRACE = 8


def fit_prices():
    """Non-negative least squares (2 params, closed form + boundary cases) per model on
    the merged TRAIN matrices. Returns {model: (p_in, p_out, med_rel_err, n)}."""
    rows = []
    for t in ("A", "B"):
        rows += [json.loads(l) for l in open(f"{DATA}/matrix_tenant{t}_train.jsonl")]
    out = {}
    for m in FLEET:
        pt = [r["prompt_tokens"] / 1e6 for r in rows]
        ct = [r["completion_tokens"] / 1e6 for r in rows]
        c = [r["cost"][m] for r in rows]
        sxx = sum(x * x for x in pt)
        syy = sum(y * y for y in ct)
        sxy = sum(x * y for x, y in zip(pt, ct))
        sxc = sum(x * z for x, z in zip(pt, c))
        syc = sum(y * z for y, z in zip(ct, c))
        det = sxx * syy - sxy * sxy
        p_in = (sxc * syy - syc * sxy) / det
        p_out = (syc * sxx - sxc * sxy) / det
        if p_in < 0:  # boundary: all cost attributed to output tokens
            p_in, p_out = 0.0, syc / syy
        if p_out < 0:
            p_out, p_in = 0.0, sxc / sxx
        errs = sorted(abs(p_in * x + p_out * y - z) / z for x, y, z in zip(pt, ct, c) if z > 0)
        out[m] = (p_in, p_out, errs[len(errs) // 2] if errs else 0.0, len(rows))
    return out


def measure_saturation(card_text):
    """Fraction of Tenant-A merged-val requests that receive >= 1 premium 429 under the
    reference router, outages disabled. The reference sends all Tenant-A traffic (gpt-5)
    to prime, value does not stock it, so every 429 an A-request sees is a prime
    throttle."""
    tmp = OUT_CARD + ".measuring"
    open(tmp, "w").write(card_text)
    try:
        from evaluator.benchmark.replay_tenants import TenantReplay
        from evaluator.reference_router import ReferenceRouter

        class Recorder(ReferenceRouter):
            def __init__(self):
                self.throttled_a = set()

            def decide(self, req, now_ms, fleet_view):
                self._cur = req
                return super().decide(req, now_ms, fleet_view)

            def on_error(self, req, now_ms, kind, retry_after_ms):
                if kind == "429" and req.features.get("tenant") == "A":
                    self.throttled_a.add(req.req_id)

        rec = Recorder()
        res = TenantReplay(tmp, split="none").run(VAL_TRACE, rec)
        n_a = res["tenants"]["A"]["requests"]
        frac = len(rec.throttled_a) / n_a
        return frac, res
    finally:
        os.remove(tmp)


def build_card(prices, saturation_comment):
    def entry(m, mult, decode_override):
        p_in, p_out, _, _ = prices[m]
        extra = f", decode_tps: {decode_override}" if decode_override else ""
        return (
            f"{{in: {p_in * mult:.4f}, out: {p_out * mult:.4f}, "
            f"cached: {p_in * mult:.4f}{extra}}}"
        )

    def provider(name, comment, models, mult):
        t = TOPO[name]
        width = max(len(m) for m in models) + 1
        lines = [f"  {name}:{' ' * max(1, 12 - len(name))}# {comment}", "    models:"]
        for m in models:
            dec = t["reasoning_decode_tps"] if m in REASONING else None
            lines.append(f"      {(m + ':').ljust(width)} {entry(m, mult, dec)}")
        for key in (
            "rpm",
            "tpm",
            "concurrency",
            "ttft_base_ms",
            "prefill_tps",
            "decode_tps",
            "cache_speedup",
            "cache_entries",
            "error_rate",
        ):
            lines.append(f"    {key}: {t[key]}")
        return "\n".join(lines)

    ladder = "\n".join(
        f"#     {m:32s} in {prices[m][0]:9.4f}  out {prices[m][1]:9.4f}  "
        f"med|rel err| {prices[m][2] * 100:5.1f}%  (n={prices[m][3]})"
        for m in sorted(FLEET, key=lambda k: -(prices[k][0] + prices[k][1]))
    )

    header = f"""# Environment card, the two-tenant shared fleet (task.md).
# Generated by evaluator/benchmark/data/make_env_card.py; FROZEN at first scored replay.
# Replayed by evaluator/benchmark/replay_tenants.py (outages, per-model
# decode, per-tenant books); prices are USD per 1M tokens, *_tps tokens/sec, windows 60 s.
#
# Price derivation: effective rates fitted on measured dollars, train only.
# Each model's (in, out) is the non-negative least-squares fit of its measured
# per-request cost on the trace's token counts over the merged TRAIN matrices
# (matrix_tenant{{A,B}}_train.jsonl, n=4839):
#     measured_cost ~= in * prompt_tokens/1e6 + out * completion_tokens/1e6
# where prompt/completion tokens are the FLEET-MEDIAN measured counts the traces carry
# (one request carries one token count; the harness bills with exactly these fields).
# The rates are therefore EFFECTIVE prices that absorb per-model verbosity: a reasoning
# model emitting 4x the median completion tokens shows ~4x the effective out-rate, so
# the harness's bill tracks the measured dollars per request and the fleet's cost
# GEOMETRY (which model is dear on which traffic) is preserved. They are not list
# prices and are not claimed to be. Fit table (base rates before reseller discount):
{ladder}
#
# CACHED == IN on every entry: the prefix-cache axis is switched off entirely
# (every trace row has prefix_tokens: 0), the operational axes under test are
# placement, rate limits, SLOs and outages, not cache warmth (rounds 1-2 covered that).
#
# LATENCY MODEL (modeled, disclosed): per-provider ttft_base_ms + prompt/prefill_tps;
# decode at the provider's decode_tps, EXCEPT the reasoning-class models
# ({", ".join(REASONING)}),
# which carry a slower per-model `decode_tps` override on every provider (long chains
# of thought decode more tokens per answer and stream slower end-to-end).
#
# PROVIDERS (modeled tiers; order matters, routers that take the first stocking
# provider get `value` for open-weight models and `prime` for the closed ones):
#   value: open-weight vendor cloud: stocks the 8 open models at the derived base
#             rates, big windows, mid latency. Tenant B's default (deepseek-r1) home.
#   prime: premium multi-model gateway: stocks all 12, fastest ttft/prefill/decode,
#             but the TIGHTEST windows (rpm {TOPO['prime']['rpm']}/min, concurrency
#             {TOPO['prime']['concurrency']}), sized so Tenant A's bursts genuinely
#             saturate it (measurement below). Scheduled outages hit this provider.
#   courier: discount reseller: all 12 at {COURIER_DISCOUNT:.0%} of the derived base
#             rates (modeled reseller margin, the one deliberate departure from the
#             fitted rates, disclosed here), slow ttft/prefill/decode, flakier. The
#             cheap deadline-tolerant overflow path, and the failover during prime
#             outages.
#
# Premium saturation, measured rather than assumed (generator embeds the measurement):
{saturation_comment}
#
# OUTAGES: two scheduled windows for `prime` on the VAL schedule, literal below.
# TEST windows exist but are DERIVED AT REPLAY TIME from test_seed by
# evaluator/benchmark/replay_tenants.py:derive_test_outage_windows, generated, never inspected by this
# round's tuning code; the harness owner sees them first at sealed-eval time.
# While a provider is down its fleet view publishes an EMPTY catalogue plus
# `outage_until`; dispatching into an announced outage more than retry_grace times
# loses the request with a typed `outage_lost` violation.
"""
    body = "\n".join(
        [
            "providers:",
            provider(
                "value",
                "open-weight vendor cloud: 8 open models, derived base rates",
                OPEN_WEIGHT,
                1.0,
            ),
            provider(
                "prime",
                "premium multi-model gateway: all 12, fastest, tightest windows, outages",
                FLEET,
                1.0,
            ),
            provider(
                "courier",
                f"discount reseller: all 12 at {COURIER_DISCOUNT:.0%} of base, slow, flaky",
                FLEET,
                COURIER_DISCOUNT,
            ),
            "retry_after_ms: 2000        # providers ask this much back-off on 429",
            "sim_seed: 11                # provider-side randomness (error injection) is seeded",
            "outages:",
            "  provider: prime",
            f"  retry_grace: {RETRY_GRACE}          # outage-blind dispatches before the request is lost",
            "  val:",
        ]
        + [f"    - [{s}, {e}]" for s, e in VAL_OUTAGES]
        + [
            f"  test_seed: {TEST_SEED}      # test windows derived at replay time, never inspected",
            "",
        ]
    )
    return header + body


def main():
    prices = fit_prices()
    placeholder = "#   (measuring...)"
    frac, res = measure_saturation(build_card(prices, placeholder))
    a = res["tenants"]["A"]
    sat = (
        f"#   Reference router, merged VAL trace, outages disabled: {frac * 100:.1f}% of "
        f"Tenant-A requests\n"
        f"#   (n_A={a['requests']}) received >= 1 prime 429; prime throttles_429 total = "
        f"{res['leading_indicators']['prime']['throttles_429']}.\n"
        f"#   Target band 10-20%: bursts must genuinely contend, steady traffic must not "
        f"starve."
    )
    text = build_card(prices, sat)
    open(OUT_CARD, "w").write(text)
    sha = hashlib.sha256(text.encode()).hexdigest()
    print(f"wrote {OUT_CARD}")
    print(f"  card sha256 {sha}")
    print(f"  A-request premium-429 fraction (val, outages off): {frac * 100:.2f}%")
    print(
        "  reference val (outages off): cost $%s, tenants %s"
        % (res["total_cost_usd"], json.dumps(res["tenants"], indent=2))
    )


if __name__ == "__main__":
    main()
