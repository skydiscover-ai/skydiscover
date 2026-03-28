# ADRS: AI-Driven Research for Systems

This directory contains the systems optimization benchmarks from the **AI-Driven Research for Systems (ADRS)** initiative at UC Berkeley.

ADRS investigates how AI — large language models, evolutionary algorithms, and multi-agent architectures — can autonomously design, optimize, and evaluate computer systems. Instead of treating systems research as a purely manual process, ADRS frames it as a closed-loop optimization problem: propose candidate algorithms, evaluate them against system-level objectives, analyze failure modes, adapt the search strategy, and iterate.

Each benchmark below defines a concrete systems task with a provided evaluator, initial program, and configuration. Solutions are evolved using SkyDiscover's evolutionary search loop.

## Benchmarks

### Cloudcast — Multi-Cloud Data Transfer

**Directory:** `cloudcast/`

Given a network of cloud regions with heterogeneous egress pricing and bandwidth, broadcast a dataset from a source region to multiple destinations at minimum total cost. The evolved algorithm must construct routing topologies (e.g., relay trees, Steiner-like structures) that exploit shared intermediate hops across transfers.

### Expert Parallelism Load Balancer (EPLB)

**Directory:** `eplb/`

In Mixture-of-Experts (MoE) model inference, a small subset of experts handles each token, leading to GPU load imbalance when certain experts become disproportionately popular. This task evolves an algorithm that decides how many replicas each expert should have and how to assign them across GPUs, optimizing both load-balance quality and rebalancing runtime.

### Model Placement (Prism)

**Directory:** `prism/`

Assign multiple LLM models to a fixed GPU cluster (80 GB per GPU) such that the worst-case KV-cache pressure ratio across GPUs is minimized. Lower pressure means more memory headroom for serving, improving throughput and stability under varying request loads.

### LLM-SQL — Column Reordering for Prefix Caching

**Directory:** `llm_sql/`

When rows of a table are serialized into LLM prompts sequentially, consecutive rows that share leading column values can reuse cached prefixes. This task evolves a column-reordering strategy that maximizes prefix-cache hit rates across multiple real-world datasets without altering the underlying data.

### Transaction Scheduling (TXN)

**Directory:** `txn_scheduling/`

Given a set of database transactions with read/write dependencies on shared keys, find an execution ordering that minimizes the total makespan. The evolved scheduler must respect conflict constraints (read-write and write-write on the same key) while compressing the overall completion time.

### Telemetry Repair

**Coming soon.** The Telemetry Repair benchmark is under active development and will be released in a future update. 

## Quick Start

Each benchmark directory contains:
- `initial_program.py` — the seed solution for evolution
- `eval/` — containerized evaluator directory
- `config.yaml` — run configuration

Run any benchmark from the repo root:

```bash
uv run skydiscover-run \
  benchmarks/ADRS/cloudcast/initial_program.py \
  benchmarks/ADRS/cloudcast/eval \
  -c benchmarks/ADRS/cloudcast/config.yaml \
  -s [your_algorithm] \
  -i 100
```

See the individual benchmark directories for task-specific setup instructions (e.g., dataset downloads, GPU dependencies).
