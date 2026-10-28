# Profiler Interpretation

## 1. Establish the measurement boundary

Before reading a trace, answer:

- Does the measurement include model loading, input preparation, compilation, warmup, synchronization, or only steady-state execution?
- Is the reported value host wall time, device kernel time, an operator aggregate, or end-to-end latency?
- Are the trace and compiler artifacts from the same run, device, mode, and shape specialization?
- Was the profiler itself active during the reported benchmark, and could it perturb timings?

Never compare numbers with different boundaries as if they were the same metric.

## 2. Common evidence sources

Depending on the stack, inspect PyTorch profiler traces/tables, Kineto/Chrome trace JSON, Nsight Systems/Compute for CUDA, or CANN `msprof` output such as summary/trace/task/operator data or a MindStudio-compatible export. Filenames, schemas, and columns vary with tool and release. Discover the actual files and consult the matching tool version instead of assuming every run emits the same set.

For each source, record whether its time is inclusive or exclusive, host or device, per invocation or aggregated, and whether synchronization is included. Keep the raw evidence available for audit; summarize only the required fields.

## 3. Correlate profile events with generated code

Use a multi-signal match:

1. Kernel/operator name or generated-code symbol.
2. Input/output shape, dtype, and device.
3. Launch order and neighboring events.
4. Stream and timestamp range where available.
5. Source location or stack metadata when enabled.

Names can be truncated, mangled, or reused. A textual name match alone is weak evidence. Build a table such as:

| Generated call/kernel | Profile event | Calls | Device time | Match confidence | Notes |
| --- | --- | ---: | ---: | --- | --- |

Explicitly mark unmatched generated kernels and unmatched profile events. The former may be conditional/dead/not exercised; the latter may be runtime, library, copy, synchronization, or non-Inductor work.

## 4. Interpret timing without overclaiming

- **Kernel count** helps explain launch overhead but is not a performance score.
- **Device duration** attributes measured device work only within the profiler's timing semantics; it may not sum to E2E latency.
- **Host gaps** can indicate launch, Python, synchronization, allocator, or scheduling overhead; verify with a timeline.
- **Memory bandwidth/occupancy/utilization** are useful only when collected with a compatible tool and interpreted for the target architecture.
- **OP aggregate** may be a benchmark-specific aggregation and can include or omit calls differently from per-kernel trace time. Do not call it “average operator latency” unless the data definition supports that.
- **E2E latency** is the model-level benchmark measurement. It can include overheads absent from kernel-only profiles.

Do not add values from different clocks or nested inclusive intervals. Avoid dividing aggregate time by a guessed operation count.

## 5. Lightweight versus detailed profiling

Lightweight modes often retain enough summary data to identify major operator/kernel costs while omitting some timeline, hardware-counter, stack, or compiler-correlation details. State what was collected and what cannot be concluded. Do not call a profile “full” or MindStudio-importable unless its emitted artifacts and schema have been verified.

Profiler overhead can change latency, launch behavior, and memory use. Use unprofiled benchmark repeats for final performance claims; use profiled runs for attribution, ideally with matching configuration and a separate recorded run ID.

## 6. Minimum evidence for a bottleneck claim

A defensible claim should include:

1. A model-level timing regression or improvement measured outside the attribution trace.
2. Profile events showing the affected runtime region and its contribution.
3. A code/IR explanation for why those events correspond to the suspected work.
4. A control experiment that changes the suspected factor while keeping workload and measurement policy fixed.

Without item 4, phrase the result as a likely cause, not a confirmed root cause.
