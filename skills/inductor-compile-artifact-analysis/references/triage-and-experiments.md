# Triage and Experiment Design

## Decision tree

### A. Correctness fails

1. Confirm the eager reference itself is stable and valid; compare repeated eager runs if relevant.
2. Record exact checker, dtype, tolerance, seed, determinism, input signature, and whether failure is forward or backward.
3. Determine whether the failing value comes from a graph break, compiled graph, external operator, graph replay, or mixed execution.
4. Reduce the failing graph region while preserving shape, layout, dtype, and relevant state.
5. Compare intermediate outputs, not only the final loss/output. Use compiler accuracy dumps or per-op comparison if available.
6. Do not relax tolerance or alter model behavior without a justified contract and explicit approval.

### B. Compilation fails

1. Capture the first meaningful exception and compiler/backend versions.
2. Identify the stage: Dynamo capture, AOTAutograd, Inductor lowering/scheduling, code generation, Triton/backend compilation, or runtime launch.
3. Find the smallest FX/IR node and generated-code context connected to the exception.
4. Check whether the API/module/IR field exists in the exact PyTorch version; distinguish an upstream API drift from a backend lowering gap.
5. Verify with an isolated minimal compile before proposing a broad compatibility change.

### C. Unexpected graph break or eager fallback

1. Inspect graph-break/recompilation logs and graph segment boundaries.
2. Check guards and input signatures across calls; dynamic dimensions and Python control flow can create new specializations.
3. Identify whether the fallback is intentional or unsupported for this backend.
4. Confirm the runtime trace contains the expected compiled calls; a captured graph alone is not sufficient.

### D. Expected fusion is absent

1. Compare readable and transformed FX to see whether nodes are still present and connected.
2. Inspect IR dependencies, layouts, mutation/alias constraints, reductions, and scheduler groups.
3. Check whether an external call or decomposition explains the boundary.
4. Quantify the unfused region's runtime share before recommending a fusion change.

### E. Performance regresses

1. Re-run or validate the unprofiled baseline; check variance and measurement boundary.
2. Exclude compile/warmup/cache effects if the target is steady state.
3. Use the trace to localize the extra time: device kernel, more launches, host overhead, copies, synchronization, or external library.
4. Tie the event to generated code/IR and quantify its share.
5. Change one factor at a time. Keep a control run and record rollback instructions.

## Good experiment design

Every experiment should state:

- Hypothesis and the evidence that motivated it.
- One changed variable; all other run-contract fields held fixed.
- Expected result if the hypothesis is right and expected result if it is wrong.
- Correctness gate, warmup/repeat policy, and artifacts to collect.
- Reversibility/cleanup, especially for caches, environment variables, or shared devices.

Prefer a paired A/B run on the same device and environment. If device contention, thermal/power state, or background jobs differ, record it and avoid strong causal claims.

## Evidence language

- **Observed:** directly present in logs, graph, IR, code, profile, or repeated measurements.
- **Inferred:** best explanation connecting multiple observations, but not uniquely proven.
- **Unverified:** plausible idea not yet tested or missing decisive evidence.
- **Root cause confirmed:** a controlled intervention changes the predicted evidence and resolves/reproduces the behavior.

## Compact findings table

| Finding | Evidence | Confidence | Impact | Next check |
| --- | --- | --- | --- | --- |
