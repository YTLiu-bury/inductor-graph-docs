# Weekly 154 模型流水清单与特化说明

## 1. 口径与来源

本文以 173 服务器上实际执行的 weekly 计划为准：

- 模型目录：`<WORKSPACE_ROOT>/workspace_benchmark/config/models.yaml`
- 集合定义：`<WORKSPACE_ROOT>/workspace_benchmark/config/model_sets.yaml`
- 实际计划：`<WORKSPACE_ROOT>/workspace_benchmark/runs/20261010-weekly-full-dvm-triton-experimental-dynamic-off/plan.json`
- 本次流水：`20261010-weekly-full-dvm-triton-experimental-dynamic-off`

`daily-full` 使用 `daily ∪ community` 的模型集合。目录中共有 161 个候选模型，其中 7 个被明确禁用，因此本次 weekly 实际执行 154 个模型。

## 2. 流水参数

| 项目 | 实际配置 |
|---|---|
| 模型数量 | 154 |
| 后端 | DVM、Triton Experimental |
| Dynamic Shape | `off` |
| ACLGraph | `disable-aclgraph` |
| 执行模式 | 默认 train + inference；按模型特化配置覆盖 |
| 总 case 数 | 524 |
| 每个后端 case 数 | 262 |
| train case | 222 |
| inference case | 302 |
| iterations | 50 |
| repeat | 1 |
| 默认 timeout | 2000 秒 |
| 特化 timeout | 3600 秒 |
| 精度检查 | 开启 `accuracy` |
| 冷启动 | 开启 `cold-start-latency` |
| 编译耗时 | 开启 `dump-compile-time` |
| 算子数据 | 开启轻量 profiler |

本次 524 个 case 的构成为：108 个模型同时跑 train 和 inference，43 个模型只跑 inference，3 个模型只跑 train。每个模型会在 DVM 和 Triton Experimental 两个后端分别执行。

## 3. 实际模型清单

### 3.1 HuggingFace：51 个

```text
AlbertForMaskedLM
AlbertForQuestionAnswering
BartForCausalLM
BartForConditionalGeneration
BertForMaskedLM
BertForQuestionAnswering
AllenaiLongformerBase
BlenderbotForCausalLM
BlenderbotSmallForCausalLM
BlenderbotSmallForConditionalGeneration
DebertaForMaskedLM
DebertaForQuestionAnswering
DebertaV2ForMaskedLM
DebertaV2ForQuestionAnswering
DistillGPT2
DistilBertForMaskedLM
DistilBertForQuestionAnswering
ElectraForCausalLM
ElectraForQuestionAnswering
GoogleFnet
GPT2ForSequenceClassification
GPTJForCausalLM
GPTJForQuestionAnswering
GPTNeoForCausalLM
GPTNeoForSequenceClassification
LayoutLMForMaskedLM
LayoutLMForSequenceClassification
M2M100ForConditionalGeneration
MBartForCausalLM
MT5ForConditionalGeneration
MegatronBertForCausalLM
MegatronBertForQuestionAnswering
MobileBertForMaskedLM
MobileBertForQuestionAnswering
OPTForCausalLM
PLBartForCausalLM
PegasusForCausalLM
PegasusForConditionalGeneration
RobertaForCausalLM
RobertaForQuestionAnswering
Speech2Text2ForCausalLM
T5ForConditionalGeneration
T5Small
TrOCRForCausalLM
XGLMForCausalLM
XLNetLMHeadModel
meta-llama/Llama-3.2-1B
Qwen/Qwen3-0.6B
YituTechConvBert
google/gemma-2-2b
openai/whisper-tiny
```

### 3.2 TIMM：18 个

```text
adv_inception_v3
beit_base_patch16_224
convnextv2_nano.fcmae_ft_in22k_in1k
deit_base_distilled_patch16_224
deit_tiny_patch16_224.fb_in1k
dm_nfnet_f0
ghostnet_100
inception_v3
mobilenetv2_100
mobilenetv3_large_100
mobilevit_s
nfnet_l0
repvgg_a2
tf_efficientnet_b0
visformer_small
vit_base_patch16_siglip_256
swin_base_patch4_window7_224
vit_base_patch14_dinov2.lvd142m
```

### 3.3 TorchBench：85 个

```text
alexnet
Background_Matting
basic_gnn_edgecnn
basic_gnn_gcn
basic_gnn_gin
basic_gnn_sage
BERT_pytorch
cm3leon_generate
DALLE2_pytorch
dcgan
densenet121
demucs
detectron2_fasterrcnn_r_101_c4
detectron2_fasterrcnn_r_101_dc5
detectron2_fasterrcnn_r_101_fpn
detectron2_fasterrcnn_r_50_c4
detectron2_fasterrcnn_r_50_dc5
detectron2_fasterrcnn_r_50_fpn
detectron2_fcos_r_50_fpn
detectron2_maskrcnn_r_101_c4
detectron2_maskrcnn_r_101_fpn
detectron2_maskrcnn_r_50_c4
detectron2_maskrcnn_r_50_fpn
dlrm
drq
fastNLP_Bert
functorch_dp_cifar10
functorch_maml_omniglot
hf_Albert
hf_Bart
hf_Bert
hf_Bert_large
hf_BigBird
hf_DistilBert
hf_distil_whisper
hf_GPT2_large
hf_T5_base
hf_T5_large
LearningToPaint
lennard_jones
mobilenet_v2
mobilenet_v2_quantized_qat
nvidia_deeprecommender
opacus_cifar10
phlippe_densenet
phlippe_resnet
pytorch_stargan
pytorch_unet
resnet152
resnet18
resnet50
resnext50_32x4d
sam
shufflenet_v2_x1_0
soft_actor_critic
speech_transformer
squeezenet1_1
timm_resnest
timm_vovnet
torch_multimodal_clip
tts_angular
vgg16
yolov3
llama
llama_v2_7b_16h
llava
maml
maml_omniglot
microbench_unbacked_tolist_sum
nanogpt
pyhpc_equation_of_state
pyhpc_isoneutral_mixing
pyhpc_turbulent_kinetic_energy
mnasnet1_0
mobilenet_v3_large
moco
moondream
resnet50_quantized_qat
Super_SloMo
timm_efficientnet
timm_vision_transformer_large
vision_maskrcnn
doctr_det_predictor
doctr_reco_predictor
sam_fast
```

## 4. 模型特化

### 4.1 只跑 inference 的模型：43 个

这些模型在配置中显式使用 `inference`，不执行训练路径：

```text
torchbench/cm3leon_generate
torchbench/DALLE2_pytorch
torchbench/detectron2_fasterrcnn_r_101_c4
torchbench/detectron2_fasterrcnn_r_101_dc5
torchbench/detectron2_fasterrcnn_r_101_fpn
torchbench/detectron2_fasterrcnn_r_50_c4
torchbench/detectron2_fasterrcnn_r_50_dc5
torchbench/detectron2_fasterrcnn_r_50_fpn
torchbench/detectron2_fcos_r_50_fpn
torchbench/detectron2_maskrcnn_r_101_c4
torchbench/detectron2_maskrcnn_r_101_fpn
torchbench/detectron2_maskrcnn_r_50_c4
torchbench/detectron2_maskrcnn_r_50_fpn
torchbench/functorch_dp_cifar10
torchbench/hf_distil_whisper
torchbench/opacus_cifar10
torchbench/sam
torchbench/yolov3
torchbench/llama_v2_7b_16h
torchbench/llava
torchbench/maml
torchbench/microbench_unbacked_tolist_sum
torchbench/pyhpc_equation_of_state
torchbench/pyhpc_isoneutral_mixing
torchbench/pyhpc_turbulent_kinetic_energy
torchbench/moondream
torchbench/doctr_det_predictor
torchbench/doctr_reco_predictor
torchbench/sam_fast
huggingface/M2M100ForConditionalGeneration
huggingface/meta-llama/Llama-3.2-1B
huggingface/Qwen/Qwen3-0.6B
huggingface/google/gemma-2-2b
huggingface/openai/whisper-tiny
timm/beit_base_patch16_224
timm/convnextv2_nano.fcmae_ft_in22k_in1k
timm/deit_base_distilled_patch16_224
timm/dm_nfnet_f0
timm/inception_v3
timm/mobilenetv2_100
timm/repvgg_a2
timm/swin_base_patch4_window7_224
timm/vit_base_patch14_dinov2.lvd142m
```

### 4.2 只跑 train 的模型：3 个

```text
torchbench/resnext50_32x4d
torchbench/mnasnet1_0
torchbench/Super_SloMo
```

### 4.3 batch size 特化

| 模型 | batch size |
|---|---:|
| huggingface/AllenaiLongformerBase | 4 |
| huggingface/BlenderbotForCausalLM | 4 |
| huggingface/GPTJForCausalLM | 1 |
| huggingface/GPTJForQuestionAnswering | 1 |
| huggingface/T5ForConditionalGeneration | 4 |
| huggingface/T5Small | 4 |
| huggingface/XLNetLMHeadModel | 8 |
| huggingface/meta-llama/Llama-3.2-1B | 1 |
| huggingface/Qwen/Qwen3-0.6B | 1 |
| huggingface/google/gemma-2-2b | 1 |

### 4.4 AMP 特化

以下模型额外开启 `amp`：

```text
torchbench/DALLE2_pytorch
torchbench/pyhpc_turbulent_kinetic_energy
torchbench/Super_SloMo
torchbench/doctr_det_predictor
torchbench/doctr_reco_predictor
torchbench/sam_fast
```

### 4.5 cosine 精度比较

以下模型额外使用 `cosine` 精度比较方式：

```text
torchbench/functorch_dp_cifar10
torchbench/opacus_cifar10
torchbench/resnext50_32x4d
torchbench/mnasnet1_0
timm/convnextv2_nano.fcmae_ft_in22k_in1k
timm/deit_base_distilled_patch16_224
timm/dm_nfnet_f0
timm/inception_v3
timm/mobilenetv2_100
timm/swin_base_patch4_window7_224
timm/vit_base_patch14_dinov2.lvd142m
```

### 4.6 timeout 特化

以下模型的单 case timeout 从默认 2000 秒提高到 3600 秒：

```text
torchbench/detectron2_fasterrcnn_r_101_dc5
torchbench/detectron2_maskrcnn_r_101_c4
torchbench/detectron2_maskrcnn_r_50_fpn
torchbench/mobilenet_v2_quantized_qat
torchbench/pytorch_unet
torchbench/resnet50_quantized_qat
torchbench/Super_SloMo
```

## 5. 明确排除的 7 个候选模型

这些模型仍在 `models.yaml` 中保留记录，但 `enabled: false`，因此不计入 154 个实际执行模型：

| Suite | 模型 | 排除原因 |
|---|---|---|
| TorchBench | `detectron2_maskrcnn` | 社区 `torchbench.yaml` 标记 `skip.all`，参考环境 OOM |
| TorchBench | `timm_regnet` | 社区配置跳过重复的 TorchBench TIMM 条目 |
| TorchBench | `timm_vision_transformer` | 社区配置跳过重复的 TorchBench TIMM 条目 |
| TorchBench | `timm_nfnet` | 社区配置跳过重复的 TorchBench TIMM 条目 |
| TorchBench | `tacotron2` | 社区 `torchbench.yaml` 标记 `skip.all`，参考环境 timeout |
| HuggingFace | `BlenderbotForConditionalGeneration` | 社区配置标记 `skip.all`，模型初始化/精度不支持 |
| HuggingFace | `mistralai/Mistral-7B-Instruct-v0.3` | 社区配置标记 `skip.all`，模型规模超出参考环境承载能力 |

## 6. 结论

当前 weekly 全量口径是 **154 个启用模型**，覆盖 HuggingFace 51、TIMM 18、TorchBench 85；使用 DVM 和 Triton Experimental 两个后端，Dynamic Shape 关闭，ACLGraph 关闭，并按照社区配置对特殊模型执行 train-only、inference-only、batch size、AMP、cosine 和 timeout 特化。

本清单描述的是 `20261010` weekly 实际生成的计划，不代表未来配置永远不变。后续修改 `config/models.yaml` 或同步社区配置后，应重新生成 `plan.json` 并更新本文档中的日期和统计数字。
