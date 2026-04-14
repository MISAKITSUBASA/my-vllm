# SPDX-License-Identifier: Apache-2.0
"""Run baseline vs popularity-aware LoRA throughput benchmarks.

This is a lightweight wrapper around ``vllm.benchmarks.throughput`` for
reproducible skewed-popularity evaluation.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path


def build_cmd(
    *,
    model: str,
    tokenizer: str,
    lora_path: str,
    max_loras: int,
    max_cpu_loras: int,
    lora_hot_ids: str | None,
    lora_assignment: str,
    hot_lora_count: int,
    hot_request_ratio: float,
    num_prompts: int,
    input_len: int,
    output_len: int,
    lora_assignment_file: str | None,
    python_executable: str,
    extra_args: list[str],
) -> list[str]:
    cmd = [
        python_executable,
        "-m",
        "vllm.benchmarks.throughput",
        "--backend",
        "vllm",
        "--dataset-name",
        "random",
        "--model",
        model,
        "--tokenizer",
        tokenizer,
        "--enable-lora",
        "--lora-path",
        lora_path,
        "--max-loras",
        str(max_loras),
        "--max-cpu-loras",
        str(max_cpu_loras),
        "--lora-assignment",
        lora_assignment,
        "--hot-lora-count",
        str(hot_lora_count),
        "--hot-request-ratio",
        str(hot_request_ratio),
        "--num-prompts",
        str(num_prompts),
        "--random-input-len",
        str(input_len),
        "--random-output-len",
        str(output_len),
    ]
    if lora_hot_ids:
        cmd += ["--lora-hot-ids", lora_hot_ids]
    if lora_assignment_file:
        cmd += ["--lora-assignment-file", lora_assignment_file]
    return cmd + extra_args


def run_cmd(cmd: list[str], output_json: Path) -> dict:
    full_cmd = cmd + ["--output-json", str(output_json)]
    subprocess.run(full_cmd, check=True)
    return json.loads(output_json.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--lora-path", required=True)
    parser.add_argument("--max-loras", type=int, default=8)
    parser.add_argument("--max-cpu-loras", type=int, default=32)
    parser.add_argument("--hot-lora-count", type=int, default=2)
    parser.add_argument("--hot-request-ratio", type=float, default=0.9)
    parser.add_argument("--num-prompts", type=int, default=2000)
    parser.add_argument("--input-len", type=int, default=256)
    parser.add_argument("--output-len", type=int, default=64)
    parser.add_argument("--output-dir", default="./benchmark_outputs")
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
        help=(
            "Python executable used to launch throughput runs. Defaults to the "
            "current interpreter, so conda/venv workflows are supported."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--lora-assignment-file",
        default=None,
        help=(
            "Optional pre-generated JSON assignment sequence. If omitted, "
            "the script generates a deterministic skewed sequence and reuses "
            "it for baseline/policy runs."
        ),
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Additional arg to forward to throughput benchmark (repeatable).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.lora_assignment_file is not None:
        assignment_file = Path(args.lora_assignment_file)
    else:
        assignment_file = output_dir / "lora_assignment_sequence.json"
        rng = random.Random(args.seed)
        hot_count = max(1, min(args.hot_lora_count, args.max_loras))
        sequence: list[int] = []
        for _ in range(args.num_prompts):
            if rng.random() < args.hot_request_ratio:
                sequence.append(rng.randint(1, hot_count))
            else:
                cold_count = args.max_loras - hot_count
                if cold_count > 0:
                    sequence.append(rng.randint(hot_count + 1, args.max_loras))
                else:
                    sequence.append(rng.randint(1, hot_count))
        assignment_file.write_text(json.dumps(sequence))

    baseline_cmd = build_cmd(
        model=args.model,
        tokenizer=args.tokenizer,
        lora_path=args.lora_path,
        max_loras=args.max_loras,
        max_cpu_loras=args.max_cpu_loras,
        lora_hot_ids=None,
        lora_assignment="skewed",
        hot_lora_count=args.hot_lora_count,
        hot_request_ratio=args.hot_request_ratio,
        num_prompts=args.num_prompts,
        input_len=args.input_len,
        output_len=args.output_len,
        lora_assignment_file=str(assignment_file),
        python_executable=args.python_executable,
        extra_args=args.extra_arg,
    )

    policy_hot_ids = ",".join(str(i) for i in range(1, args.hot_lora_count + 1))
    policy_cmd = build_cmd(
        model=args.model,
        tokenizer=args.tokenizer,
        lora_path=args.lora_path,
        max_loras=args.max_loras,
        max_cpu_loras=args.max_cpu_loras,
        lora_hot_ids=policy_hot_ids,
        lora_assignment="skewed",
        hot_lora_count=args.hot_lora_count,
        hot_request_ratio=args.hot_request_ratio,
        num_prompts=args.num_prompts,
        input_len=args.input_len,
        output_len=args.output_len,
        lora_assignment_file=str(assignment_file),
        python_executable=args.python_executable,
        extra_args=args.extra_arg,
    )

    baseline_json = output_dir / "baseline.json"
    policy_json = output_dir / "popularity_aware.json"

    baseline_res = run_cmd(baseline_cmd, baseline_json)
    policy_res = run_cmd(policy_cmd, policy_json)

    summary = {
        "baseline": baseline_res,
        "popularity_aware": policy_res,
        "delta_requests_per_second": (
            policy_res["requests_per_second"] - baseline_res["requests_per_second"]
        ),
        "delta_tokens_per_second": (
            policy_res["tokens_per_second"] - baseline_res["tokens_per_second"]
        ),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote benchmark outputs to {output_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
