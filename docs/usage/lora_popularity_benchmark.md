# Popularity-Aware Multi-LoRA Benchmark Guide

## Goal

Compare baseline multi-LoRA runtime behavior against popularity-aware residency policy under skewed adapter demand.

## What was added

- Skewed LoRA request assignment in throughput benchmark:
  - `--lora-assignment skewed`
  - `--hot-lora-count`
  - `--hot-request-ratio`
  - `--lora-assignment-file` for fixed, reusable request-to-LoRA sequence
- Convenience runner:
  - `vllm/benchmarks/lora_popularity_benchmark.py`

## Reproducible benchmark commands

### 1) Run side-by-side baseline vs popularity-aware

```bash
python -m vllm.benchmarks.lora_popularity_benchmark \
  --model <base_model> \
  --tokenizer <tokenizer> \
  --lora-path <path_or_hf_lora> \
  --max-loras 8 \
  --max-cpu-loras 32 \
  --hot-lora-count 2 \
  --hot-request-ratio 0.9 \
  --num-prompts 2000 \
  --input-len 256 \
  --output-len 64 \
  --seed 0
```

This writes:

- `benchmark_outputs/baseline.json`
- `benchmark_outputs/popularity_aware.json`
- `benchmark_outputs/summary.json`
- `benchmark_outputs/lora_assignment_sequence.json` (fixed sequence reused by both runs)

### 2) Manual single-run baseline (no hints)

```bash
python -m vllm.benchmarks.throughput \
  --backend vllm --dataset-name random \
  --model <base_model> --tokenizer <tokenizer> \
  --enable-lora --lora-path <path_or_hf_lora> \
  --max-loras 8 --max-cpu-loras 32 \
  --lora-assignment-file benchmark_outputs/lora_assignment_sequence.json \
  --lora-assignment skewed --hot-lora-count 2 --hot-request-ratio 0.9
```

### 3) Manual single-run popularity-aware

```bash
python -m vllm.benchmarks.throughput \
  --backend vllm --dataset-name random \
  --model <base_model> --tokenizer <tokenizer> \
  --enable-lora --lora-path <path_or_hf_lora> \
  --max-loras 8 --max-cpu-loras 32 \
  --lora-hot-ids 1,2 \
  --lora-assignment-file benchmark_outputs/lora_assignment_sequence.json \
  --lora-assignment skewed --hot-lora-count 2 --hot-request-ratio 0.9
```

## MacBook Pro testing notes

You can use a MacBook Pro to **run through correctness / workflow smoke tests**.

- If no CUDA/NVIDIA GPU is available, treat this as functional validation.
- Throughput/latency numbers from Mac should not be used as final GPU-serving claims.
- For final performance conclusions (throughput/TTFT/latency), rerun on your target GPU server.

## Conda note

If you use Anaconda/Miniconda, activate your env first and run the same
`python -m ...` commands. The benchmark wrapper now defaults to the current
interpreter (`sys.executable`) and can also be overridden with
`--python-executable`.

## Troubleshooting

- If wrapper reports `baseline.json` / `popularity_aware.json` missing, run the
  printed throughput command directly. This usually means the inner benchmark
  failed before writing `--output-json` (model download/auth/env mismatch).
- First run may take a long time due to Hugging Face model/adapter download and
  model initialization. For smoke tests on laptops, start with small values
  (e.g., `--num-prompts 8 --input-len 16 --output-len 8`) and scale up after
  confirming end-to-end success.

## Metrics to compare

- `requests_per_second`
- `tokens_per_second`
- (if available in your environment) TTFT and end-to-end latency from serving benchmarks.

## How to interpret

- Improvement with `--lora-hot-ids` under high `--hot-request-ratio` supports the skewed-popularity residency hypothesis.
- Little/no improvement at low skew (e.g. `--hot-request-ratio` near 0.5) suggests workload is closer to uniform demand.
- Sensitivity to `--max-loras` validates that residency pressure is a key factor.

## Known incompleteness / future work

- V1 is runtime-policy-only; no CUDA/kernel changes.
- Current policy uses static hot-ID hints; dynamic online popularity estimation is future work.
- Level 2 pre-merge-hot/cold-delta architecture is documented as future design, not implemented here.
