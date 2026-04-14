# Popularity-Aware Multi-LoRA Residency Policy (Level 1)

## Current LoRA runtime path in this repository

1. **Request carries LoRA selection** via `LoRARequest` (`lora_int_id`, `lora_path`).
2. **Scheduler enforces max active LoRAs per batch** in `vllm/v1/core/sched/scheduler.py` via `max_loras` checks.
3. **Worker builds active LoRA set per step** in `vllm/v1/worker/gpu_model_runner.py` and `vllm/v1/worker/lora_model_runner_mixin.py`, then calls `set_active_adapters`.
4. **Worker-side LoRA manager applies residency/load** in `vllm/lora/worker_manager.py` (`LRUCacheWorkerLoRAManager`) backed by `LRUCacheLoRAModelManager` in `vllm/lora/model_manager.py`.
5. **Baseline eviction behavior** is plain LRU for active GPU adapter slots when full.

## Bottleneck under skewed popularity

With hot/cold skew, plain LRU may evict frequently reused hot adapters when cold adapters appear, causing avoidable GPU residency churn and repeated re-activation overhead.

## Level 1 optimization implemented

A **popularity-aware eviction preference** was added at the worker-side LoRA cache/residency layer:

- New static hint: `hot_lora_ids` in LoRA config.
- New CLI arg: `--lora-hot-ids 1,2,3`.
- Eviction policy in `LRUCacheLoRAModelManager.activate_adapter`:
  - If hints are absent: preserve baseline LRU.
  - If hints are present and GPU slots are full: evict oldest **cold** adapter first.
  - If all active adapters are hot: fallback to LRU behavior (required for progress).

This keeps the change localized and runtime-policy-only, with no kernel/CUDA modifications.

## Insertion points and why they are the right tradeoff

Chosen insertion points:

- `vllm/config/lora.py`: config-level hint plumbing.
- `vllm/engine/arg_utils.py`: minimally invasive CLI exposure.
- `vllm/lora/model_manager.py`: GPU residency eviction decision point.

Why this is a good systems-project tradeoff:

- Small, explainable delta.
- Directly targets residency churn hypothesis.
- Preserves default behavior.
- Avoids broad scheduler or kernel refactors in V1.

## Level 2 (future) pre-merge-hot / cold-delta design

A second-stage comparison could:

1. Offline-merge one dominant hot adapter into base weights.
2. Represent cold adapters as deltas relative to that merged base.
3. Keep runtime policy to handle residual cold churn.

Pros: potentially less runtime adapter switching cost for ultra-dominant hot adapter.
Cons: operational complexity (artifact management/versioning), less flexible online updates, larger evaluation surface.

Given these tradeoffs, this repo is better served by the current Level 1 runtime policy first.
