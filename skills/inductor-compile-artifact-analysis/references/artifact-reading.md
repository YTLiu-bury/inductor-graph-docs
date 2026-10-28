# Artifact Reading and Graph Correlation

## 1. Find the right run

Do not assume one fixed `torch_compile_debug` layout. Locate the run root from the benchmark log, `TORCH_COMPILE_DEBUG` setting, or compiler output; then inventory files and timestamps. Common names include `torch_compile_debug/`, `model__*`, `torchdynamo/`, `torchinductor/`, `fx_graph_runnable.py`, `fx_graph_readable.py`, `fx_graph_transformed.py`, `ir_pre_fusion.txt`, `ir_post_fusion.txt`, and `output_code.py`. Names and nesting vary by PyTorch release and backend.

Linux examples:

```bash
find "$RUN_ROOT" -type f \( -name 'fx_graph_readable.py' -o -name 'fx_graph_transformed.py' \
  -o -name 'ir_pre_fusion.txt' -o -name 'ir_post_fusion.txt' -o -name 'output_code.py' \) -print
find "$RUN_ROOT" -type f -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %p\n' | sort
```

PowerShell examples:

```powershell
Get-ChildItem $RunRoot -Recurse -File -Include fx_graph_readable.py,fx_graph_transformed.py,ir_pre_fusion.txt,ir_post_fusion.txt,output_code.py |
  Select-Object FullName,Length,LastWriteTime
```

Use timestamps, graph IDs, model/mode directory, and the associated log to exclude stale artifacts. If a directory contains several attempts, identify the attempt that corresponds to the successful or failing run being discussed.

## 2. Build an artifact map

Create a compact inventory before interpreting content:

| Graph/attempt | Mode | Direction | FX readable | FX transformed | IR pre | IR post | Generated code | Profile match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

Forward and backward may be compiled separately and may have multiple specialization variants. Record shapes/guards for each. Do not compare a forward graph on one side with backward code on the other.

## 3. Read each layer

### `fx_graph_readable.py`

Use this as the model-level map: operations, data flow, parameters/buffers, input placeholders, and source-stack hints where present. Pick a concrete region (for example normalization -> activation -> pointwise op) and follow its values downstream. FX node count is not kernel count.

### `fx_graph_runnable.py`

Treat this as a compiler-input/reproduction artifact. It can expose concrete graph inputs and make a targeted repro easier. It is not a runtime trace and does not prove that the exact graph was executed in the measured interval.

### `fx_graph_transformed.py`

Compare node coverage and dependencies with the readable graph. Identify rewrites, decompositions, lifted constants, eliminated nodes, graph breaks, and shape-specialized branches. A node disappearing may mean folding, fusion, decomposition, or elimination; continue to IR/code before assigning meaning.

### `ir_pre_fusion.txt` and `ir_post_fusion.txt`

Compare scheduler groups and dependencies, not raw line counts. Track:

- Which pointwise/reduction/memory operations become one group.
- Where reductions, layout conversions, mutation, aliasing, or dependencies stop fusion.
- Input/output layouts, masks, index expressions, and reduction dimensions.
- Whether a group is emitted as generated code or lowered to an external library/operator.

The exact IR vocabulary is release-dependent. Check the corresponding Inductor implementation if an IR field or stage is ambiguous.

### `output_code.py`

Separate wrapper code from device code. Identify generated Triton kernels, C++/CUDA code, external calls, argument construction, launch grid, and synchronization. A generated function definition is not proof it ran; a call site is not proof it dominates latency. Count launch sites after accounting for conditional branches and repeated calls.

## 4. Trace an operation end to end

For one representative region, produce this chain:

```text
model source/input
  -> FX node(s)
  -> transformed FX node(s)
  -> scheduler/IR group
  -> generated kernel or external operator call
  -> matching profiler event(s)
```

Record names and shapes at every transition. If a mapping is ambiguous, label it as inferred and identify what would disambiguate it (debug symbols, source stacks, kernel metadata, or a narrow isolated run).

## 5. Separate compiler behavior from runtime behavior

- **Graph break / eager fallback:** graph coverage is interrupted; verify logs/graph segmentation and runtime events rather than inferring from a missing fused kernel.
- **Decomposition:** one high-level operation becomes several lower-level operations. It may improve backend support or lose a fused library path; inspect both code and timing.
- **Fusion:** fewer launches or less intermediate traffic is a structural fact; speedup is a separate measured result.
- **External library call:** generated wrapper delegates work; analyze the actual library/kernel activity when possible.
- **Specialization/recompile:** different guards or shapes can create multiple compiled variants. Match each measured invocation to the correct variant.

Do not infer implementation semantics from generated variable names alone. Use source, IR dependencies, and runtime evidence together.
