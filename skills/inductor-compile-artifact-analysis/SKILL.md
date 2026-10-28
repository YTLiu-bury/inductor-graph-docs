---
name: inductor-compile-artifact-analysis
description: Analyze PyTorch torch.compile/Inductor artifacts and runtime profiles to diagnose graph capture, fusion, generated kernels, correctness, and performance. Use for torch_compile_debug, FX graphs, Inductor IR, output_code.py, profiler traces, NPU/GPU comparisons, or explaining model-level compile performance.
---

# Inductor Compile Artifact Analysis

Use this skill to turn compiler artifacts and runtime measurements into a traceable explanation of what a model compiled, what actually ran, and where performance differences came from. It applies to PyTorch Inductor on CPU, CUDA/GPU, and PrivateUse1/NPU backends, including Triton-based backends. Backend details vary; never transfer a CUDA assumption to NPU without evidence.

## Operating Rules

- Start from the user's question: correctness, graph coverage, fusion, kernel quality, runtime attribution, or a fair device comparison. Do not inspect every artifact if a smaller evidence chain answers it.
- Treat each run as an experiment. Record exact revisions, environment, inputs, mode, compiler settings, device, and run identity before comparing files.
- Keep compilation structure, generated code, runtime execution, and end-to-end timing as separate evidence layers. A source file shows what was generated, not what executed or how much time it took.
- Pair forward and backward independently. Pair artifacts from the same compile attempt, graph, model mode, and input signature; stale or neighboring directories are not substitutes.
- Separate correctness from performance. Do not interpret a faster but correctness-failing run as a valid optimization. Record the comparison criterion and tolerance.
- Label conclusions as **Observed**, **Inferred**, or **Unverified**. A plausible mechanism is not a root cause until a discriminating experiment supports it.
- Do not edit models, compiler settings, caches, or benchmark configuration unless asked. Prefer read-only inspection and controlled, reversible experiments.
- Redact usernames, hostnames, private paths, tokens, proxy details, model inputs, and other sensitive material from shareable reports.

## Workflow

1. **Scope the question.** Define the target model/case and whether the task is correctness, compilation, graph coverage, fusion, runtime, or NPU-vs-GPU performance.
2. **Establish provenance.** Capture the run contract below. Mark missing fields rather than guessing.
3. **Inventory and pair artifacts.** Identify run root, graph directories, forward/backward graphs, compile attempts, and profiler outputs. Build a graph-to-kernel-to-profile mapping where possible.
4. **Read artifacts in compiler order.** Start with readable FX, compare transformed FX, inspect pre/post-fusion IR, then inspect generated wrapper and device code. Follow one important operation path end to end.
5. **Check runtime evidence.** Match profiler events to generated call sites using names, shapes, launch order, and timestamps. Distinguish device execution from host overhead, synchronization, compilation, and data loading.
6. **Compare baselines under controlled conditions.** Normalize workload and measurement policy. Explain graph/fusion scope separately from kernel strategy and hardware/runtime differences.
7. **Test the explanation.** Propose the smallest experiment that distinguishes competing causes; do not present an untested hypothesis as a fix.
8. **Report reproducibly.** Include evidence paths (redacted where needed), concise tables, limitations, confidence, and the next action.

## Run Contract

Record as many of these as are available:

| Area | Fields |
| --- | --- |
| Workload | Model/revision, suite, train or inference, forward/backward, input source, batch and full shape signature |
| Numerics | dtype/autocast, layout/strides, seeds, determinism settings, correctness checker and tolerance |
| Compiler | PyTorch revision/version, torch_npu if applicable, Inductor options, backend, Triton/compiler revision, dynamic/static, graph/capture settings |
| Runtime | Device model/count, driver/runtime/CANN or CUDA versions, CPU/thread settings, environment variables relevant to compilation |
| Measurement | Warmup, repeats/iterations, synchronization, measured region, profiler mode, cache state |
| Identity | Timestamp, run ID, log path, compile-debug root, profile root |

If two runs differ in a field that can affect graph capture, compilation, numerics, or timing, disclose it and narrow the comparison.

## Progressive References

Read only the references needed for the task:

- [Artifact reading and graph correlation](references/artifact-reading.md): discovery, pairing, FX/IR/code interpretation, and graph-to-kernel mapping.
- [Profiler interpretation](references/profiler-interpretation.md): trace formats, timing boundaries, kernel attribution, and common metric traps.
- [NPU/GPU comparison](references/npu-gpu-comparison.md): controlled comparison, different backend strategies, and a report template.
- [Triage and experiment design](references/triage-and-experiments.md): decision tree for correctness, graph breaks, fusion, compile errors, and performance gaps.

Useful upstream starting points include the [PyTorch Inductor debugging tutorial](https://docs.pytorch.org/tutorials/intermediate/inductor_debug_cpu.html), [torch.compile troubleshooting](https://docs.pytorch.org/docs/stable/torch.compiler_troubleshooting.html), and [torch.profiler documentation](https://docs.pytorch.org/docs/stable/profiler.html). These are general PyTorch references; verify backend-specific behavior against the exact version and backend source.
