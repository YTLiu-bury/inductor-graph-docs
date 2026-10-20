# Community 106 YAML Notes

This note explains `auto_board/run_community_106.yaml`.

## Scope

`auto_board/run_community_106.yaml` contains only the 106 community dashboard models:

- `Bench`: 58 TorchBench models
- `HuggingFace`: 30 HuggingFace models
- `Timm`: 18 TIMM models

It intentionally excludes internal extra models from `auto_board/run.yaml`, such as old `hf_*` TorchBench aliases, old `timm_*` TorchBench aliases, extra detectron2 variants, and quantized QAT models.

## Inheritance

The file inherits the current default execution style from `auto_board/run.yaml`:

- `backend: inductor`
- `iterations: 50`
- `--accuracy`
- `--cold-start-latency`
- `--train`
- `--dump-compile-time`
- `--enable-profiler`
- `--disable-aclgraph`

Existing per-model overrides were kept when the model is part of the 106 list.

## Inference Overrides

The current BenchBoard YAML runner can directly express train/inference mode by replacing the section `flags`.

The following models are set to inference in `run_community_106.yaml`:

- Existing inherited overrides:
  - `cm3leon_generate`
  - `detectron2_fcos_r_50_fpn`
  - `maml`
  - `pyhpc_equation_of_state`
  - `yolov3`
- Upstream or source-code confirmed inference-only / training-not-supported:
  - `doctr_det_predictor`
  - `doctr_reco_predictor`
  - `llama_v2_7b_16h`
  - `llava`
  - `microbench_unbacked_tolist_sum`
  - `moondream`
  - `pyhpc_isoneutral_mixing`
  - `pyhpc_turbulent_kinetic_energy`
  - `sam`
  - `sam_fast`
  - `M2M100ForConditionalGeneration`

## Rules Not Yet Expressible In Current YAML

Upstream benchmark YAML contains several rules that the current BenchBoard `run_model.py` cannot fully represent as first-class config:

- `batch_size.training` / `batch_size.divisors`
- tolerance buckets such as `higher`, `even_higher`, `higher_bf16`
- `only_fp32`, e.g. `GoogleFnet`
- TIMM `emulate_precision_casts`, relevant to models such as `mobilenetv2_100`
- repeated accuracy checks, e.g. `sam` / `sam_fast`

These are not encoded as arbitrary args because passing unsupported CLI options would break current runners. They should be added after the runner grows explicit support.

## High-Risk Models

These models are included because they are part of the community 106, but need attention before expecting green results:

- `moco`: upstream model code only supports CUDA/XLA-style distributed paths and raises `NotImplementedError` for other devices. YAML flags alone cannot fix NPU.
- `Qwen/Qwen3-0.6B`, `google/gemma-2-2b`, `meta-llama/Llama-3.2-1B`, `openai/whisper-tiny`: require model/data/cache availability and may hit auth or download constraints.
- `doctr_det_predictor`, `doctr_reco_predictor`: inference-only; also require doctr dependencies.
- `sam`, `sam_fast`: inference-only and have upstream special accuracy handling.
- `llava`, `moondream`: inference-only multimodal models; likely need model/cache/data preparation.

