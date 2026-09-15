"""Shared-prefix generation benchmark for an OpenAI-compatible /v1/completions server.

A few long shared prefixes, each branched into many sampled candidates (Qwen3-4B on one L4:
2 roots x 24k-token prefix x 32 branches x 128 tokens). The score is aggregate generated tokens
per second of decode wall-clock; everything else printed is diagnostic. Prompts are sent as token
ids so the prefix is byte-identical across branches and engines, and the prefix cache is warmed
before timing. Use the same client and flags for every engine being compared.

    python benchmark.py --url http://localhost:8000 --model Qwen/Qwen3-4B
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass

import httpx

DEFAULT_MODEL = "Qwen/Qwen3-4B"


@dataclass
class BranchResult:
    root: int
    branch: int
    ttft_s: float | None  # time to first generated token (cache hot -> small)
    total_s: float  # end-to-end for this branch (dispatch -> done)
    output_tokens: int
    decode_tps: float  # this branch's own decode rate
    error: str | None


# ---------------------------------------------------------------------------
# Workload construction (deterministic, reproducible, list[int] prompts)
# ---------------------------------------------------------------------------
def build_workload(
    roots: int,
    branches: int,
    prefix_len: int,
    vocab_size: int,
    shared_seed: int,
    branch_seed_base: int,
) -> list[tuple[int, int, list[int], int]]:
    """Return [(root, branch, prompt_token_ids, sampling_seed), ...]: one fixed prompt per root, shared
    by all its branches, each branch with its own sampling seed.
    """
    lo, hi = 100, max(101, vocab_size - 100)
    work: list[tuple[int, int, list[int], int]] = []
    for r in range(roots):
        rng = random.Random(shared_seed + r)
        prompt = [rng.randint(lo, hi) for _ in range(prefix_len)]
        for b in range(branches):
            seed = branch_seed_base + r * branches + b  # distinct per branch
            work.append((r, b, prompt, seed))
    return work


def load_vocab_size(model: str) -> int:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "transformers is required for tokenizer vocab: pip install transformers"
        ) from exc
    return AutoTokenizer.from_pretrained(model).vocab_size


# ---------------------------------------------------------------------------
# One streaming request
# ---------------------------------------------------------------------------
async def run_branch(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    prompt_ids: list[int],
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
    root: int,
    branch: int,
) -> BranchResult:
    # Extended sampling params (top_k / seed / ignore_eos) are understood by vLLM and
    # SGLang's /v1/completions; unknown fields are ignored by compliant servers. Our
    # engine must honor the same contract.
    body = {
        "model": model,
        "prompt": prompt_ids,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "seed": seed,
        "ignore_eos": True,  # emit exactly max_tokens -> clean throughput
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.perf_counter()
    t_first: float | None = None
    out_tokens = 0
    error: str | None = None
    try:
        async with client.stream(
            "POST",
            f"{url}/v1/completions",
            json=body,
            headers={"content-type": "application/json"},
            timeout=httpx.Timeout(connect=30, read=1200, write=120, pool=30),
        ) as resp:
            if resp.status_code != 200:
                raw = await resp.aread()
                raise RuntimeError(
                    f"http {resp.status_code}: {raw.decode('utf-8', 'replace')[:400]}"
                )
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data = line[len("data: ") :]
                if data == "[DONE]":
                    break
                try:
                    ev = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = ev.get("choices") or []
                if choices and choices[0].get("text"):
                    if t_first is None:
                        t_first = time.perf_counter()
                    out_tokens += 1  # chunk fallback count
                usage = ev.get("usage")
                if usage and usage.get("completion_tokens") is not None:
                    out_tokens = usage["completion_tokens"]  # authoritative if given
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    t_end = time.perf_counter()
    ttft = (t_first - t0) if t_first is not None else None
    decode_window = max(t_end - (t_first or t0), 1e-6)
    return BranchResult(
        root=root,
        branch=branch,
        ttft_s=ttft,
        total_s=t_end - t0,
        output_tokens=out_tokens,
        decode_tps=(out_tokens / decode_window) if out_tokens else 0.0,
        error=error,
    )


# ---------------------------------------------------------------------------
# Percentile helper
# ---------------------------------------------------------------------------
def pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = (len(s) - 1) * (p / 100.0)
    lo, hi = math.floor(k), math.ceil(k)
    return s[int(k)] if lo == hi else s[lo] * (hi - k) + s[hi] * (k - lo)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
async def run(args: argparse.Namespace) -> dict:
    print(f"[bench] tokenizer vocab for {args.model}", file=sys.stderr)
    vocab = load_vocab_size(args.model)
    work = build_workload(
        args.roots,
        args.branches,
        args.prefix_len,
        vocab,
        args.shared_seed,
        args.seed,
    )
    n = len(work)
    print(
        f"[bench] {args.roots} roots x {args.branches} branches = {n} sequences | "
        f"prefix={args.prefix_len} out={args.max_tokens} "
        f"sampling(t={args.temperature},p={args.top_p},k={args.top_k})",
        file=sys.stderr,
    )

    base = args.url.rstrip("/")
    cap = max(n * 2, 128)
    limits = httpx.Limits(max_connections=cap, max_keepalive_connections=cap)
    async with httpx.AsyncClient(limits=limits) as client:
        # WARM each root's shared prefix (so the measured window is decode)
        if args.warmup:
            warm_t0 = time.perf_counter()
            warm = await asyncio.gather(
                *[
                    run_branch(
                        client,
                        base,
                        args.model,
                        work[r * args.branches][2],
                        max_tokens=min(8, args.max_tokens),
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        seed=args.seed - 1 - r,
                        root=r,
                        branch=-1,
                    )
                    for r in range(args.roots)
                ]
            )
            warm_ttft = [w.ttft_s for w in warm if w.ttft_s is not None]
            if warm_ttft:
                print(
                    f"[warmup] {args.roots} prefixes in {time.perf_counter()-warm_t0:.2f}s "
                    f"(prefill ttft mean={statistics.mean(warm_ttft):.2f}s)",
                    file=sys.stderr,
                )
            else:
                print("[warmup] done", file=sys.stderr)

        # MEASURED burst: all branches concurrently; time this window only
        print(f"[bench] dispatching {n} concurrent branches ...", file=sys.stderr)
        t_run0 = time.perf_counter()
        results: list[BranchResult] = await asyncio.gather(
            *[
                run_branch(
                    client,
                    base,
                    args.model,
                    prompt,
                    args.max_tokens,
                    args.temperature,
                    args.top_p,
                    args.top_k,
                    seed,
                    root,
                    branch,
                )
                for (root, branch, prompt, seed) in work
            ]
        )
        wall = time.perf_counter() - t_run0

    ok = [r for r in results if r.error is None]
    bad = [r for r in results if r.error is not None]
    total_out = sum(r.output_tokens for r in ok)
    aggregate = total_out / wall if wall > 0 else 0.0

    ttfts = [r.ttft_s for r in ok if r.ttft_s is not None]
    totals = [r.total_s for r in ok]
    decs = [r.decode_tps for r in ok if r.decode_tps > 0]

    # report
    print()
    print("=" * 64)
    print("  Hierarchical shared-prefix generation benchmark")
    print("=" * 64)
    print(f"Backend:            {base}/v1/completions   model={args.model}")
    print(f"Workload:           {args.roots} roots x {args.branches} branches = {n} seqs")
    print(f"Prefix / output:    {args.prefix_len} shared / {args.max_tokens} generated")
    print(
        f"Sampling:           temp={args.temperature} top_p={args.top_p} top_k={args.top_k} (per-branch seeds)"
    )
    print(f"Completed / failed: {len(ok)} / {len(bad)}")
    print(f"Wall clock:         {wall:.2f}s   total_output_tokens={total_out}")
    if ttfts:
        print(
            f"TTFT (s)   p50/p95/p99: {pct(ttfts,50):.3f} / {pct(ttfts,95):.3f} / {pct(ttfts,99):.3f}  "
            f"(small => prefix cache hot)"
        )
    if totals:
        print(
            f"Latency(s) p50/p95/p99: {pct(totals,50):.2f} / {pct(totals,95):.2f} / {pct(totals,99):.2f}"
        )
    if decs:
        print(f"Per-branch decode tps  mean/p50: {statistics.mean(decs):.2f} / {pct(decs,50):.2f}")
    print()
    print(f"aggregate_throughput_tok_per_sec = {aggregate:.2f}")
    print()
    if bad:
        print("Errors (first 3):")
        for r in bad[:3]:
            print(f"  root={r.root} branch={r.branch}: {r.error[:160]}")

    result = {
        "config": {
            "url": f"{base}/v1/completions",
            "model": args.model,
            "roots": args.roots,
            "branches": args.branches,
            "sequences": n,
            "prefix_len": args.prefix_len,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "shared_seed": args.shared_seed,
            "seed": args.seed,
            "warmup": bool(args.warmup),
        },
        "aggregate_throughput_tok_per_sec": aggregate,
        "num_completed": len(ok),
        "num_failed": len(bad),
        "wall_clock_sec": wall,
        "total_output_tokens": total_out,
        "ttft_sec": (
            {"p50": pct(ttfts, 50), "p95": pct(ttfts, 95), "p99": pct(ttfts, 99)} if ttfts else None
        ),
        "latency_sec": (
            {"p50": pct(totals, 50), "p95": pct(totals, 95), "p99": pct(totals, 99)}
            if totals
            else None
        ),
        "decode_tps_per_branch": (
            {"mean": statistics.mean(decs), "p50": pct(decs, 50)} if decs else None
        ),
        "per_branch": [asdict(r) for r in results],
    }
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results written to {args.output_json}")
    return result


def main() -> None:
    p = argparse.ArgumentParser(
        description="Hierarchical shared-prefix (Hydragen) throughput benchmark."
    )
    p.add_argument("--url", default="http://localhost:8000", help="server base URL")
    p.add_argument("--model", default=DEFAULT_MODEL, help="model id (tokenizer + completions name)")
    p.add_argument(
        "--roots",
        type=int,
        default=2,
        help="independent shared-prefix roots (>=2 defeats global-prefix cascade)",
    )
    p.add_argument("--branches", type=int, default=32, help="candidates per root")
    p.add_argument("--prefix-len", type=int, default=24576, help="shared prefix length in tokens")
    p.add_argument(
        "--max-tokens", type=int, default=128, help="generated tokens per candidate (128 or 256)"
    )
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument(
        "--seed", type=int, default=1000, help="base seed; each branch gets a distinct offset"
    )
    p.add_argument(
        "--shared-seed", type=int, default=0, help="seed for the deterministic shared prefixes"
    )
    p.add_argument(
        "--warmup", type=int, default=1, help="warm each root's prefix before measuring (1=yes)"
    )
    p.add_argument("--output-json", type=str, default=None)
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
