# NPU/GPU Comparison

## 1. Define the comparison question

Separate these questions; they require different evidence:

1. **Eager vs compiled on one device:** what changed in correctness, graph coverage, runtime, or memory?
2. **NPU vs GPU model performance:** how does the same workload perform end to end?
3. **Fusion coverage:** which graph regions fuse on each backend, and where are boundaries different?
4. **Same work, different kernel quality:** for comparable work, which implementation/launch strategy is faster and why?

Do not use an answer to one question as proof of another. For example, NPU compile beating NPU eager does not establish parity with GPU.

## 2. Match the workload and declare mismatches

Match, or explicitly disclose differences in:

- Model code/revision, weights, input values, batch/sequence/image shapes, and train/inference path.
- Forward/backward coverage, optimizer behavior, loss, and output collection.
- Dtype/autocast, numerical mode, determinism, and correctness criterion.
- Compiler/backend versions and options, dynamic-shape behavior, graph/capture settings, and warmup/compile-cache state.
- Number of devices, CPU threads, synchronization, iteration count, repeats, and measured region.

Hardware-specific graph or library strategies need not be identical. The requirement is a matched semantic workload and an honest description of strategy differences.

## 3. Analyze in three layers

### Layer A: Model-level result

Compare steady-state E2E latency/throughput and variability using the same measurement contract. Include correctness status. For training, distinguish forward, backward, optimizer, and complete-step timing if the harness exposes them.

### Layer B: Work/fusion coverage

For each backend, map high-level operations to compiled regions, fused scheduler groups, generated kernels, and external calls. Make a concise region table:

| Region | NPU lowering | GPU lowering | Structural difference | Runtime evidence |
| --- | --- | --- | --- | --- |

“More fused” is not automatically faster. A backend may intentionally use a vendor library call instead of generated Triton, or preserve a fusion boundary for layout/reduction/resource reasons.

### Layer C: Kernel/runtime efficiency

For matched or semantically equivalent regions, compare algorithm, tile/block choices, memory access/coalescing, vectorization, reduction strategy, temporary traffic, launch count, occupancy/resource pressure, and measured device time. Use architecture-appropriate tools; raw counters are not directly comparable across unlike devices.

## 4. Normalize carefully

- Prefer raw latency and throughput alongside any ratio.
- State ratio direction explicitly, e.g. `GPU latency / NPU latency`; values above one then mean NPU is faster under that definition.
- Do not average ratios without explaining weighting. Geometric mean is often suitable for multiplicative speedups, while arithmetic means answer a different question.
- Report per-model values and sample counts; avoid hiding missing or failed cases in an aggregate.
- Separate compilation time from steady-state time unless the use case includes startup.
- Use repeated runs and report median plus spread (or another declared statistic). Do not rely on one noisy sample.

## 5. Interpret differing strategies

When NPU and GPU lowerings differ, use this order:

1. Verify semantic and shape equivalence.
2. Identify intentional backend-specific lowering (library, decomposition, unsupported path, graph break).
3. Quantify how much of the model runtime the differing region consumes.
4. Compare only comparable work; use a microbenchmark only if it preserves the relevant shape/layout/dtype and avoids changing the algorithm.
5. Decide whether the gap is a fusion opportunity, kernel-quality issue, library gap, launch/synchronization overhead, or workload mismatch.

Do not force identical fusion just to make graphs look alike. The goal is equivalent work and a useful performance explanation, not identical source.

## 6. Report template

```markdown
## Scope
- Model/case and question:
- NPU/GPU device and software revisions:
- Workload and measurement boundary:
- Correctness criterion/result:

## Results
| Device/backend | E2E latency | Throughput | Runs/statistic | Notes |
| --- | ---: | ---: | --- | --- |

## Structural comparison
| Model region | NPU graph/lowering | GPU graph/lowering | Profile contribution | Interpretation |
| --- | --- | --- | ---: | --- |

## Findings
- **Observed:** evidence and artifact/profile paths.
- **Inferred:** explanation supported by the evidence.
- **Unverified:** competing hypotheses and missing evidence.

## Next experiment
- Single controlled change, expected discriminating outcomes, and rollback plan.
```
