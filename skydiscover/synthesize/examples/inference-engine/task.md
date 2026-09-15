# Task: hierarchical shared-prefix generation for Qwen3-4B on a single L4

Build an inference engine that maximises aggregate output throughput (tok/s) on a heavily branched,
long shared-prefix generation workload while staying accurate, and serve it behind an
OpenAI-compatible `v1/completions` server. Build from scratch.

- Model: `Qwen/Qwen3-4B`, a GQA model with 36 layers, 8 KV heads, `head_dim=128`.
- Precision: BF16 weights, BF16 KV cache, non-thinking mode.
- Allowed: Triton, CUDA, cuBLASLt / CUTLASS / FlashInfer, tokenizer and checkpoint loading.
  PyTorch only for validation, not as the production engine.

## Workload
The benchmark, `evaluator/benchmark.py`, sends 64 concurrent requests by default: 2
independent roots x 32 branches per root. Each root is a fixed 24,576-token prefix shared by all
32 of its branches; each branch samples a distinct output sequence of 128 tokens by default (256 in
the longer variant). So the server handles 64 concurrent sequences over 2 shared prefixes.

This is the only workload that matters for performance. Optimize for this shape, not for arbitrary
serving, but the server must still serve arbitrary requests.

## Score
- Metric: aggregate output throughput, the number the benchmark prints.
- Baseline: a fairly configured SGLang, with prefix cache / RadixAttention warmed, CUDA graphs on,
  tuned batching, and identical model, precision, and workload. On this L4 it reaches about
  272 tok/s at 128 output tokens per branch and about 271 tok/s at 256.
- Beating the baseline is the floor, not the goal; keep driving throughput toward the hardware
  limit.

## Testing
Correctness is judged on greedy decoding against the HuggingFace transformers reference for this
model. The performance workload is sampled (temperature 0.7 and so on, from the benchmark);
correctness is a separate greedy check.

- Fixed prompts: for the prompts in `evaluator/accuracy_prompts.json`, greedy output must
  be token-exact against the HF reference. These prompts were chosen so greedy is well defined; any
  mismatch is a real bug. `single_prompts` run one at a time. Each of `shared_prefix_sets` runs all
  its branches together as prefix + branch sharing the prefix, and each branch is compared to HF run
  on that prefix + branch alone; a match shows the shared-prefix path is exact. 24 to 48 greedy
  tokens per prompt is enough.
- Other prompts: output must be close to HF, exact on confident tokens, with only occasional
  divergence on near-ties (bf16 rounding on flat or degenerate logits, where the HF reference itself
  does not give the same answer every time). Byte-identical output on arbitrary inputs is neither
  required nor well defined.
- The scored path: the path that is actually timed, sampled, with a ~24k prefix and 32 branches,
  must itself be correct. Passing the greedy checks above is not enough; a fast wrong sampled path
  behind a correct greedy one is a reward hack. Specifically:
  - On a real, confident ~24k prefix with several branches, greedy output must match HF per branch.
    This shows the full prefix is attended and branches are not shared, and catches prefix
    truncation and branch replication.
  - At temperature 0 the sampled path must reduce to the greedy path (same forward pass). With fixed
    seeds the 32 branches must be reproducible and genuinely distinct samples, not copies.
  - Reported output tokens must be real decode steps, with no count inflation.
