# KernelBench Integration with SkyDiscover

GPU kernel optimization tasks using the [KernelBench](https://github.com/ScalingIntelligence/KernelBench) dataset and evaluation protocol.

## Overview

The KernelBench integration allows you to run SkyDiscover on any problem from the KernelBench dataset. The framework automatically:

1. Fetches the reference implementation of the target kernel from KernelBench
2. Creates an initial_program.py with EVOLVE-BLOCK markers
3. Configures the evaluator with problem-specific parameters
4. Runs the optimization using either a containerized or native Python evaluator

The evaluator uses the KernelBench evaluation infrastructure to measure speedup over PyTorch eager execution.

### Evaluator Modes

- **Containerized (Docker)**: Runs evaluation inside a Docker container (default)
- **Native Python**: Runs evaluation directly as Python code (for clusters without Docker/Podman)

## Directory Structure

```
benchmarks/kernelbench/
├── config.yaml              # System prompt + search/evaluator settings
├── resolver.py              # Benchmark loader (fetches target problems from KernelBench)
├── requirements.txt         # Resolver dependencies (kernelbench library)
└── evaluator/               # Self-contained Docker benchmark
    ├── Dockerfile           # Container image definition
    ├── evaluate.sh          # Entrypoint (receives solution path)
    ├── evaluator.py         # Scoring logic using KernelBench
    ├── requirements.txt     # Evaluator dependencies (kernelbench[gpu])
    └── wrapper.py           # JSON protocol wrapper
```

**Note:** The `run_and_check.py` script is downloaded directly from the KernelBench repository during Docker build (pinned to commit `423217d` for reproducibility). To update, modify the `KERNELBENCH_COMMIT` build arg in the Dockerfile.

## Installation

Before using the KernelBench integration, install the required dependencies:

```bash
# Install KernelBench library (required for problem fetching)
uv pip install -r benchmarks/kernelbench/requirements.txt
```

**Note:** The resolver (problem fetching) only needs the base `kernelbench` package. The containerized evaluator installs `kernelbench[gpu]` for GPU support.

## Quick Start

### Using Docker (Default)

Edit `benchmarks/kernelbench/config.yaml` to select a target kernel from the [KernelBench database](https://huggingface.co/datasets/ScalingIntelligence/KernelBench):

```yaml
benchmark:
  # KernelBench problem specification
  level: 2                    # Problem difficulty level (1, 2, 3 or 4)
  problem_id: 5               # Specific problem ID within the level
```

Then, run optimization on this problem:

```bash
# algo can be "adaevolve", "evox", "topk", "beam_search", "best_of_n", etc.
uv run skydiscover-run benchmarks/kernelbench/evaluator/ \
  -c benchmarks/kernelbench/config.yaml \
  --search <algo> \
  --iterations 50
```

### Using Native Python (No Docker Required)

For clusters without Docker/Podman privileges, you can run the evaluator as native Python code.

#### 1. Install Dependencies

```bash
# Install KernelBench with GPU support
pip install -r benchmarks/kernelbench/evaluator/requirements.txt
```

#### 2. Configure Native Mode

Edit `benchmarks/kernelbench/config.yaml`:

```yaml
benchmark:
  enabled: true
  name: kernelbench
  resolver: benchmarks.kernelbench.resolver
  
  # Set to false to use native Python evaluator (no Docker)
  use_docker: false
  
  level: 2
  problem_id: 11
  # ... rest of config
```

#### 3. Run Optimization

```bash
# algo can be "adaevolve", "evox", "topk", "beam_search", "best_of_n", etc.
uv run skydiscover-run benchmarks/kernelbench/evaluator/ \
  -c benchmarks/kernelbench/config.yaml \
  --search <algo> \
  --iterations 50
```

**Note:** The `run_and_check.py` script from KernelBench will be automatically downloaded on first run.

**Note:** No initial_program argument is needed - it is fetched automatically based on the `benchmark` section in config.yaml.

## Configuration Reference

### Benchmark Section

The `benchmark` section in `config.yaml` controls problem loading:

```yaml
benchmark:
  enabled: true                    # Enable benchmark loader
  name: kernelbench                # Benchmark name (for logging)
  resolver: benchmarks.kernelbench.resolver  # Python module path
  
  # Evaluator mode
  use_docker: true                 # true: containerized (Docker), false: native Python
  
  # Problem specification
  level: 1                         # Difficulty: 1 (easy), 2 (medium), 3 (hard), 4 (very hard)
  problem_id: 1                    # Problem ID within the level
  
  # Dataset source
  dataset_src: huggingface         # 'huggingface' or 'local'
  dataset_name: ScalingIntelligence/KernelBench  # HF dataset name
  
  # Evaluation settings
  eval_mode: local                 # 'local' or 'modal'
  gpu: H100                        # GPU type: H100, A100, etc.
  num_correct_trials: 5            # Correctness validation runs
  num_perf_trials: 100             # Performance measurement runs
```

### Environment Variables

The resolver provides these environment variables to the evaluator:

- `KERNELBENCH_LEVEL`: Problem difficulty level (1, 2, or 3)
- `KERNELBENCH_PROBLEM_ID`: Specific problem within the level
- `KERNELBENCH_EVAL_MODE`: Evaluation mode (local, modal)
- `KERNELBENCH_GPU`: GPU type (H100, A100, etc.)
- `KERNELBENCH_NUM_CORRECT_TRIALS`: Number of correctness validation runs
- `KERNELBENCH_NUM_PERF_TRIALS`: Number of performance measurement runs
- `KERNELBENCH_TIMEOUT`: Timeout per evaluation in seconds

These variables are passed directly to the evaluator (not set globally), ensuring isolation between concurrent runs.

### Evaluation Modes

- **local**: Run evaluation on your local machine (requires GPU)
- **modal**: Run evaluation on Modal's cloud GPUs (requires Modal setup)

### GPU Types

The list of currently supported GPU types can be found [here](https://github.com/ScalingIntelligence/KernelBench/blob/423217d9fda91e0c2d67e4a43bf62f96f6d104f1/scripts/run_and_check.py#L16).

## Metrics

The evaluator returns:

- **combined_score**: Speedup over PyTorch eager execution (primary metric)
- **speedup_over_eager**: Same as combined_score
- **speedup_over_compile**: Speedup over torch.compile()
- **kernel_time_ms**: Execution time of optimized kernel
- **ref_eager_time_ms**: Reference eager execution time


## Traditional Usage (Manual Initial Program)

You can still provide an initial program manually if needed:

```bash
# Run with explicit initial program
uv run skydiscover-run my_kernel.py benchmarks/kernelbench/evaluator/ \
  -c benchmarks/kernelbench/config.yaml \
  --search <algo>
```

## Troubleshooting

### Error: "kernelbench package not found"

Install KernelBench:
```bash
pip install "kernelbench[gpu] @ git+https://github.com/ScalingIntelligence/KernelBench.git"
```

### Error: "Failed to resolve benchmark problem"

Check that:
1. `benchmark.enabled` is `true` in config
2. `level` and `problem_id` are valid
3. KernelBench package is installed
4. You have internet access (for HuggingFace dataset)

### Generated Files Location

The framework creates temporary files in `/tmp/skydiscover_kernelbench_*/`:
- `initial_program.py`: Generated initial program
- Evaluator uses the existing `benchmarks/kernelbench/evaluator/` directory
