# NPUSim BCResNet Golden Validation Guide

This document defines a practical, layered validation methodology for Coral NPU IP
using the BCResNet int8 flow in `tests/npusim_examples`.

## 1. What is the golden reference here?

For this project flow, the **software golden** is:

- `run_bcresnet_ref.cc` (TFLM standard kernels only)
- compared against optimized kernels through `npusim_run_bcresnet.py`

Reference runner registration (no custom optimized registration):

- `AddConv2D()`
- `AddDepthwiseConv2D()`
- other model ops (`Reshape`, `Pad`, `Transpose`, `Mul`, `Add`, `Sum`, etc.)

Optimized runner registration:

- `AddConv2D(Register_CONV_2D())`
- `AddDepthwiseConv2D(Register_DEPTHWISE_CONV_2D())`

## 2. Is TFLM an 8-bit implementation?

TFLM supports multiple datatypes (for example float32/int16/int8 in conv kernels).
In this BCResNet validation flow, we specifically use an **int8 model** and
`int8` input/output buffers, so the comparison target is int8 numerical semantics.

## 3. Can TFLM be used as Coral IP golden?

Yes, as a **software semantic golden** for kernel correctness.

This means:

- It is valid for checking whether optimized kernels preserve TFLM operator
  semantics (including quantization behavior).
- It is not a replacement for RTL/microarchitectural golden checks (timing,
  protocol, micro-state, power modeling).

For int8 kernel correctness, this should be treated as a bit-exact target.

## 4. Layered validation strategy

Use all layers below instead of relying on one test.

### L0: Kernel micro tests

Purpose: fast local sanity for selected conv shapes.

Command:

`bazel test //sw/opt/litert-micro/test:conv_sim_test --nocache_test_results --test_output=summary`

Pass condition:

- test passes
- no output mismatch in test log

### L1: Isolated operator differential checks

Purpose: attribute model-level mismatch to conv/depthwise paths.

Commands:

- `bazel run //tests/npusim_examples:npusim_run_bcresnet -- --mode conv_only --num_samples 5 --seed 123 --allow_diff`
- `bazel run //tests/npusim_examples:npusim_run_bcresnet -- --mode depthwise_only --num_samples 5 --seed 123 --allow_diff`

Pass condition (int8 strict mode target):

- `mismatched_samples=0`
- `max_abs_diff=0`
- `total_l1_diff=0`

### L2: Full-network differential check

Purpose: verify end-to-end semantic equivalence under real operator interaction.

Command:

`bazel run //tests/npusim_examples:npusim_run_bcresnet -- --mode full --num_samples 5 --seed 123 --allow_diff`

Pass condition:

- `mismatched_samples=0`
- `max_abs_diff=0`
- `total_l1_diff=0`

### L3: Performance and fallback observability

Purpose: validate optimization coverage and collect perf trends.

Use `run_bcresnet.cc` log outputs:

- `PROFILE_CSV_BEGIN/END`
- `FALLBACK_SUMMARY,CONV_2D,...`
- `FALLBACK_SUMMARY,DEPTHWISE_CONV_2D,...`

Expected for optimized BCResNet int8 path:

- `CONV_2D` fallback count: 0
- `DEPTHWISE_CONV_2D` fallback count: 0

## 5. Recommended acceptance criteria (for int8 optimized kernels)

For a change to be accepted:

1. L0 passes.
2. L1 passes for `conv_only` and `depthwise_only` with zero diff.
3. L2 passes for `full` with zero diff.
4. Fallback counters remain at expected coverage (typically zero for targeted ops).

If any differential metric is non-zero, treat as a correctness bug first,
performance issue second.

## 6. Typical mismatch root causes

Common reasons for non-zero differential results:

1. Quantization mismatch in postprocess stage:
   - not using exact `MultiplyByQuantizedMultiplier` semantics
   - incorrect shift handling (for example reconstructing shift from truncated data)
2. Different clamp/order of operations vs reference.
3. Repacking/indexing mistakes in kernel inner loops.
4. Silent fallback differences between reference and optimized paths.

## 7. Practical notes

- Keep random seed fixed (`--seed`) when triaging.
- Use `--num_samples` > 1 (recommended 5+) to avoid overfitting to one sample.
- Isolated modes (`conv_only` / `depthwise_only`) should be run before full mode
  when debugging.

