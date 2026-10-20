# PyTorch 社区三套 Benchmark YAML 分析

> 分析日期：2026-10-08  
> 分析对象：PyTorch main 分支 benchmarks/dynamo 下的 TorchBench、HuggingFace、TIMM 三套 runner 配置。  
> 目的：明确社区全量模型的来源、看护标准、模型级特化方式，以及这些规则如何映射到我们的 NPU 流水。

## 1. 结论先行

PyTorch 社区的三套 benchmark 不是“把所有模型用同一条命令跑一遍”。它由四层组成：

1. **模型来源层**：TorchBench 从 pytorch/benchmark 注册表发现模型；HuggingFace 和 TIMM 从固定清单发现模型。
2. **统一运行层**：三个 runner 共用一套 common.py 参数和结果处理逻辑，分别执行 accuracy/performance、training/inference。
3. **YAML 策略层**：YAML 描述模型级差异，例如 batch size、容忍度、模式限制、设备限制、慢模型和 cudagraph 行为。
4. **CI 调度层**：GitHub Actions 负责 backend、dtype、dynamic shapes、cudagraph、分片、超时和周期间隔等全局配置。

社区的“看护标准”不是要求每个模型在每种组合下都必须通过，而是：

- 对纳入测试的模型执行统一的准确性和性能流程；
- 对不适用、超大、超时、非确定性或平台不支持的模型进行显式分类；
- 用 YAML 保留可解释的模型级例外；
- 用 workflow 分片和超时控制保证整套 benchmark 稳定完成；
- 性能结果以 eager 为基准归一化为 1x，并在 dashboard 中比较两个 commit。

对我们的 NPU 流水，最重要的原则是：**模型清单、社区标准特化、NPU 平台特化、内部调度参数分开管理**，不要把一次性的 NPU workaround 混进社区通用配置。

## 2. 官方配置和脚本入口

### 2.1 官方文件

| Suite | Runner | YAML | 模型清单入口 |
| --- | --- | --- | --- |
| TorchBench | torchbench.py | [torchbench.yaml](https://github.com/pytorch/pytorch/blob/main/benchmarks/dynamo/torchbench.yaml) | pytorch/benchmark 仓库的 TorchBench model registry |
| HuggingFace | huggingface.py | [huggingface.yaml](https://github.com/pytorch/pytorch/blob/main/benchmarks/dynamo/huggingface.yaml) | [huggingface_models_list.txt](https://github.com/pytorch/pytorch/blob/main/benchmarks/dynamo/huggingface_models_list.txt) + EXTRA_MODELS |
| TIMM | timm_models.py | [timm_models.yaml](https://github.com/pytorch/pytorch/blob/main/benchmarks/dynamo/timm_models.yaml) | [timm_models_list.txt](https://github.com/pytorch/pytorch/blob/main/benchmarks/dynamo/timm_models_list.txt) |

三套 runner 和常用参数见官方 [benchmarks/dynamo/README.md](https://github.com/pytorch/pytorch/blob/main/benchmarks/dynamo/README.md)。

### 2.2 当前 main 的清单规模

截至本文分析时：

- HuggingFace 固定清单：40 个条目；
- TIMM 固定清单：18 个条目；
- TorchBench：YAML 不直接给出全量模型列表，runner 从 pytorch/benchmark 的模型注册表读取，再根据 YAML 的 skip 等规则过滤。

这里的 40 和 18 是当前 PyTorch main 的固定清单规模，不等同于我们之前维护的 106 个社区对齐快照。仓库中的 [auto_board/run_community_106.yaml](../auto_board/run_community_106.yaml) 是一个固定的历史对齐集合，当前 main 的模型清单可能已经继续增加或调整。

### 2.3 模型如何被发现

#### TorchBench

torchbench.py 通过 torchbenchmark._list_model_paths() 获取模型名。torchbench.yaml 不负责列出所有模型，而负责：

- 跳过不适合当前设备或模式的模型；
- 指定只训练、只推理、慢模型、精度特殊处理；
- 指定 detectron2 等模型族的共同策略；
- 保留性能实验或平台兼容性标签。

TorchBench 的真实全量范围取决于 benchmark 仓库 commit、runner 代码和 YAML 三者的组合。

#### HuggingFace

huggingface.py 读取 huggingface_models_list.txt 中的模型名和默认 batch size，并额外加入代码中定义的 EXTRA_MODELS。默认 batch size 是社区在 A100 40GB 语境下预先维护的值；YAML 的 divisors 再对部分模型做缩小。

输入不是来自真实数据集，而是由 runner 根据模型类型构造随机 token、attention mask、decoder input 和 labels。不同任务类型会生成不同的 loss 输入，例如 masked LM、causal LM、sequence classification、question answering 和 conditional generation。

#### TIMM

timm_models.py 读取 timm_models_list.txt 中的模型名和默认 batch size，加载 pretrained TIMM 模型，并根据模型的 data config 生成输入。清单不是 TIMM 的全部模型，而是社区从不同模型族抽取的代表集合。

官方 runner 还提供 refresh_model_names()：按模型族抽取代表模型，避免把 TIMM 上千个模型全部纳入 dashboard。社区关注的是模型族和算子形态覆盖，而不是模型数量最大化。

## 3. 社区统一看护标准

### 3.1 正确性和性能分开

| 类型 | 目标 | 典型状态 |
| --- | --- | --- |
| --accuracy | 对比 eager 与编译后结果/梯度是否满足容忍度 | pass_accuracy、fail_accuracy、fail_to_run |
| --performance | 统计 eager 和 compiled 的稳定执行耗时及 speedup | 以 eager 为 1x 的归一化性能 |

同一个模型可能 accuracy 通过但性能没有提升，也可能能编译但 accuracy 失败。社区不会把这些状态折叠成一个 pass 字段。

### 3.2 training 和 inference 分开

官方 dashboard 同时覆盖 training 和 inference：

- training 通常配合 --amp；
- inference 通常配合 --bfloat16；
- 只支持一种模式的模型由 YAML 显式声明 only_training 或 only_inference；
- 不适合训练的模型通过 skip.test.training 排除，而不是让它在流水中无意义失败。

所以模型结果必须带 mode 信息。不能把 training pass 当成 inference pass，也不能把 only-inference 模型的训练失败统计为后端失败。

### 3.3 dtype 和编译配置显式记录

社区常用的全局 profile 包括：

- backend：通常为 inductor；
- training/inference；
- amp、bfloat16、float16、float32；
- dynamic shapes / dynamic batch；
- cudagraphs 是否启用；
- freezing、C++ wrapper、AOT Inductor、max autotune；
- cold-start latency；
- profiler 或 torch trace；
- total_partitions 和 partition_id 分片参数。

这些是一次流水的全局实验条件，不应该被写成单个模型的隐式行为。模型 YAML 只覆盖确实需要不同处理的模型。

### 3.4 设备和资源限制显式分类

官方配置把以下情况单独记录：

- 指定设备不支持；
- 特定平台超时；
- 指定显存下 OOM；
- 要求多进程或多卡；
- control flow 或 export 限制；
- eager 端本身不稳定；
- 只在特定 dtype 下可比。

这类状态进入 skip.device、skip.test、skip.export_not_supported、skip.all 等类别，而不是简单标成后端失败。

### 3.5 分片、超时和监控

当前官方 nightly workflow 使用 A100 runner，并将性能测试拆分为：

- HuggingFace：5 个 shard；
- TIMM：6 个 shard；
- TorchBench：6 个 shard；
- cachebench：2 个 shard。

常规 test timeout 为 720 分钟，weekly max-autotune 流程为 1440 分钟。workflow 还记录 monitor log 和数据采集间隔，并导出 profiler trace 和 torch trace。

这说明“单模型 timeout”和“整条流水 timeout”是两层概念。我们的流水也应同时保存：

1. 模型级开始时间、结束时间、状态和错误分类；
2. suite/shard 级执行时间和进程状态；
3. 整条流水级总耗时和是否被外部 kill。

参考：[官方 inductor-perf-test-nightly.yml](https://github.com/pytorch/pytorch/blob/main/.github/workflows/inductor-perf-test-nightly.yml)。

## 4. torchbench.yaml 逐项分析

### 4.1 batch size

当前特化：

| 作用域 | 模型和值 | 原因 |
| --- | --- | --- |
| training | demucs: 4、densenet121: 4、timm_efficientdet: 1、llama_v2_7b_16h: 1、yolov3: 8 | 控制大数据集、激活或 cudagraph OOM |
| training | dlrm: 1024 | 推荐模型需要较大 batch 才能体现 embedding/MLP 负载 |
| inference | timm_efficientdet: 32 | 单独设置推理 batch |
| dont_change_batch_size | demucs、pytorch_struct、pyhpc_turbulent_kinetic_energy、vision_maskrcnn | 禁止通用 batch 缩放逻辑再次修改 |

### 4.2 精度容忍度和比较方法

| 配置 | 当前模型 | 含义 |
| --- | --- | --- |
| higher | alexnet、attention_is_all_you_need_pytorch、densenet121、vgg16、mobilenet_v3_large、nvidia_deeprecommender | GPU 非确定性 kernel 下使用更宽容忍度 |
| even_higher | soft_actor_critic、tacotron2、yolov3、squeezenet1_1、shufflenet_v2_x1_0 | 需要高于 1e-3 |
| higher_fp16 | doctr_reco_predictor、drq、phlippe_resnet | FP16 单独放宽 |
| higher_bf16 | doctr_reco_predictor、drq | BF16 单独放宽 |
| higher_bf16_xpu | squeezenet1_1、phlippe_resnet | XPU BF16 特化 |
| freezing.even_higher | mobilenet_v2 | freezing 的 conv-batchnorm fusion 引入更大数值差异 |

以下模型使用 IoU 检查布尔 mask，而非逐元素完全相等：

- detectron2_maskrcnn_r_101_fpn
- detectron2_maskrcnn_r_50_c4
- sam
- sam_fast
- vision_maskrcnn

sam 和 sam_fast 的 IoU 阈值从默认 0.99 调整为 0.95，并各执行 5 次 accuracy check，以多数通过作为结论。这是针对离散 mask 和运行波动的比较策略，不是关闭 accuracy。

其他 accuracy 特化：

- functorch_dp_cifar10、yolov3：需要更大的小 tensor multiplier；
- mobilenet_v3_large：标记为 non_deterministic；
- Background_Matting、pytorch_unet：eager 端不能开启 deterministic algorithms；
- maml、llama_v2_7b_16h、Background_Matting、stable_diffusion_unet：过大，跳过同时保存 eager、dynamo 和 FP64 reference 的完整 accuracy 流程；
- pytorch_unet：max_batch_size 为 2。

### 4.3 training-only、设备限制和导出限制

only_training 包括：

- 全部 detectron2 模型；
- tts_angular、tacotron2、demucs；
- hf_Reformer、pytorch_struct、yolov3、modded_nanogpt。

官方注释说明这些模型不能在 eval mode 完成同等 accuracy 检查。detectron2 不能简单用 inference 跑通替代社区 training 口径。

skip.test.training 还排除：

- 非训练设计的 pyhpc_equation_of_state、pyhpc_isoneutral_mixing、pyhpc_turbulent_kinetic_energy、maml、llama、llama_v2_7b_16h、simple_gpt、sam_fast；
- 缺少默认 training batch 配置的 cm3leon_generate、doctr_det_predictor、doctr_reco_predictor、moondream；
- 资源不足的 phi_1_5、detectron2_fcos_r_50_fpn。

skip.export_not_supported 包括 doctr、drq、llama、sam_fast、soft_actor_critic、timm_efficientdet、vision_maskrcnn。这是 export 能力边界，不应记成普通 compile fail。

### 4.4 cudagraph、性能和长期不稳定模型

- tts_angular 使用 disable_cudagraph，官方说明 speedup 会在 .05 到 1.05 间波动；
- very_slow 以 i9-11900K CPU 超过 600 秒为参考；
- slow 以超过 60 秒为参考，并额外列出 BERT_pytorch、demucs、fastNLP_Bert、speech_transformer、vision_maskrcnn；
- trt_not_yet_working 是 TensorRT 标签，不代表 TorchInductor 在所有设备都失败；
- canary_models 中的 torchrec_dlrm 仍要求运行，即便它属于 canary；
- skip.all 包含已知 OOM、timeout、eager 失败、multi-GPU 不可用、重复 suite 模型、长期失效模型和 issue 关联模型。

TorchBench YAML 会跳过 HF/TIMM 重复模型，因为它们已经在独立 suite 中维护。三套 suite 的合并清单不能简单把所有名字直接相加。

## 5. huggingface.yaml 逐项分析

### 5.1 跳过策略

当前 skip.all 的原因是模型或测试框架限制：

| 模型 | 社区记录的原因 |
| --- | --- |
| Reformer | .eval() 不支持，accuracy 流程难以建立 |
| BlenderbotForConditionalGeneration、GPTNeoForCausalLM、GPTNeoForSequenceClassification | deepcopy 失败 |
| DebertaV2ForMaskedLM | 当前 TorchScript 导入 transformers.modeling_deberta_v2 失败 |
| GPTJForCausalLM、GPTJForQuestionAnswering | batch size 为 1 仍失败 |
| google/gemma-3-4b-it、openai/gpt-oss-20b、mistralai/Mistral-7B-Instruct-v0.3 | 模型过大 |

CPU skip 还包括 meta-llama/Llama-3.2-1B、google/gemma-2-2b、google/gemma-3-4b-it、openai/whisper-tiny、Qwen/Qwen3-0.6B、mistralai/Mistral-7B-Instruct-v0.3、openai/gpt-oss-20b。这不是对 CUDA/NPU 的全局 skip。

AllenaiLongformerBase 被列入 control_flow，表示它需要在控制流相关测试中单独处理。

### 5.2 batch size divisor

HF list 提供每个模型默认 batch size，YAML 的 batch_size.divisors 再将默认值除以指定倍数：

- divisor 2：AlbertForMaskedLM、AllenaiLongformerBase、BartForCausalLM、BertForMaskedLM、DebertaV2ForMaskedLM、DistilBertForMaskedLM、DistillGPT2、ElectraForCausalLM、GPT2ForSequenceClassification、GoogleFnet、LayoutLMForMaskedLM、MBartForCausalLM、MT5ForConditionalGeneration、MobileBertForMaskedLM、OPTForCausalLM、PLBartForCausalLM、RobertaForCausalLM、T5ForConditionalGeneration、T5Small、TrOCRForCausalLM、XLNetLMHeadModel、YituTechConvBert；
- divisor 4：M2M100ForConditionalGeneration、MegatronBertForCausalLM、PegasusForCausalLM；
- divisor 8：BlenderbotForCausalLM、meta-llama/Llama-3.2-1B、google/gemma-2-2b、google/gemma-3-4b-it、openai/whisper-tiny、Qwen/Qwen3-0.6B、mistralai/Mistral-7B-Instruct-v0.3、openai/gpt-oss-20b。

这些 divisor 是资源约束后的运行参数，不是模型语义要求。

### 5.3 模式、dtype 和 tolerance

- M2M100ForConditionalGeneration：only_inference，因为 training mode 下 Dynamo 失败；
- GoogleFnet：only_fp32；
- MT5ForConditionalGeneration：higher_training；
- GPT2ForSequenceClassification：higher_inference 和 higher_inference_cpu；
- DebertaV2ForMaskedLM、BlenderbotForCausalLM：accuracy 的 large_models skip。

HF runner 代码中还有两个不在 YAML 里的重要行为：

1. 普通 HF 模型如果带 config.use_cache，runner 会设置 use_cache=False，以保持这套旧 dashboard 测试口径；
2. HF LLM generation 模型使用 generation_config.disable_compile=True，并用 model.generate 测性能，避免把 generation 控制逻辑误当作普通 forward 图。

这两点属于 runner 语义，不应误写成 YAML 的模型特化。

## 6. timm_models.yaml 逐项分析

### 6.1 当前 TIMM 清单

当前固定清单为 18 个模型：

~~~~text
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
swin_base_patch4_window7_224
tf_efficientnet_b0
visformer_small
vit_base_patch14_dinov2.lvd142m
vit_base_patch16_siglip_256
~~~~

清单覆盖 CNN、Inception、EfficientNet、NFNet、MobileViT、DeiT、Swin、ViT、ConvNeXt 和 SigLIP 等形态，而不是跟随 TIMM 全量模型数量变化。

### 6.2 平台和 batch 特化

skip.device.cpu_aarch64 当前包含：

- dm_nfnet_f0
- nfnet_l0
- visformer_small

官方原因是 AArch64 timeout。这个 skip 只针对 CPU AArch64，不能解释为 GPU、XPU 或 NPU 全局不支持。

batch 特化分两类：

- ci_accuracy.vit_base_patch14_dinov2.lvd142m: 4：CUDA Inductor CI accuracy 使用更低 batch，避免 FP64 golden reference OOM；
- divisors 为 2：beit_base_patch16_224、deit_base_distilled_patch16_224、gluon_xception65、mobilevit_s、swin_base_patch4_window7_224。

### 6.3 tolerance、loss 和 precision cast

| 配置 | 当前模型 | 作用 |
| --- | --- | --- |
| higher_training | inception_v3、mobilenetv3_large_100 | training accuracy 放宽 |
| higher_fp16_xpu | botnet26t_256 | XPU FP16 放宽 |
| even_higher | deit_base_distilled_patch16_224、vit_base_patch16_siglip_256 | 需要更高 tolerance |
| highest_training | beit_base_patch16_224 | training 需要 16 * 1e-2 |
| freezing | adv_inception_v3 | freezing 的 conv-batchnorm fusion 引入数值变化 |
| scaled_compute_loss | mobilevit_s | loss 缩小 1000 倍，减少梯度检查噪声 |
| require_larger_multiplier_for_smaller_tensor | inception_v3、mobilenetv3_large_100、vit_base_patch14_dinov2.lvd142m | 小 tensor 使用更大的比较 multiplier |

emulate_precision_casts 当前包括：

- vit_base_patch14_dinov2.lvd142m
- mobilenetv2_100

官方注释说明，这用于保留 eager autocast 的低精度来回转换，避免 Inductor 融合时消除 round-trip，进而放大为梯度差异。它不是通用的“精度失败就打开”开关。

accuracy.skip.eager_not_deterministic 当前包含 mobilenetv2_100，原因是 eager 的 cuDNN conv/batchnorm backward 非确定性。它只跳过该模型的 eager deterministic accuracy 比较，不代表模型不需要跑 performance。

## 7. 社区特化与 NPU 特化的边界

### 7.1 可以直接借鉴

以下策略与设备关系较小，适合直接借鉴到 NPU 流水设计：

- 三套 suite 的模型清单和 suite 去重规则；
- training/inference 分开；
- accuracy/performance 分开；
- 模型级 batch size 和最大 batch size；
- only-training / only-inference；
- slow / very-slow / timeout 分类；
- 多次 accuracy check 和 majority pass；
- 错误原因分类，而不是只记录一个 fail；
- shard、模型级耗时和总流水耗时分层记录。

### 7.2 不能直接照搬

以下配置带有明确的 GPU、CPU 或 XPU 语义，不能未经验证直接转为 NPU 规则：

- trt_not_yet_working：TensorRT 语义，与 NPU backend 无直接关系；
- skip.device.cpu、cpu_aarch64、cuda、xpu：设备限定不能替代 NPU 支持判断；
- GPU cudagraph OOM 的 batch 调整：NPU 的 ACLGraph/NPUGraph 内存行为不同；
- GPU/XPU 专属 tolerance：NPU 需要基于 NPU eager、compiled、dtype 和后端的误差证据；
- CUDA-only 的 FBGEMM、CUDA graph、cuDNN 非确定性结论；
- 官方 dashboard 的 A100 batch size：不能直接作为 A2/A3/A5/A6 的默认 batch。

### 7.3 对 NPU 流水的推荐配置分层

建议把配置分成四层：

~~~~text
model_catalog.yaml
  社区/内部模型集合、suite、模型分类和来源

community_policy.yaml
  与 PyTorch 社区一致的 mode、dtype、batch、tolerance 和 skip 语义

npu_policy.yaml
  NPU backend、ACLGraph/NPUGraph、NPU 数据依赖、NPU 容忍度和已验证 workaround

pipeline_profile.yaml
  日期、torch/torch_npu/CANN/benchmark commit、卡分配、shard、timeout、profiler 策略
~~~~

其中：

- 模型是否纳入，由 model_catalog.yaml 决定；
- 社区是否只训练/只推理，由 community_policy.yaml 决定；
- NPU 是否使用 Triton-Experimental、DVM、ACLGraph，由 npu_policy.yaml 和流水 profile 决定；
- 日常流水的 rawdata 和 dashboard JSON 不应反过来修改社区模型口径。

## 8. 对当前 BenchBoard 的具体启示

### 8.1 不要把 106 配置当成官方 main 的永久清单

当前仓库中的 run_community_106.yaml 适合做历史外部看板快照，但官方 main 的 HuggingFace 清单已经是 40 个条目，并且有新模型和新 skip 规则。建议同时维护：

- community_snapshot_106：保持外部看板历史口径稳定；
- community_main_current：定期从官方三个清单/YAML 同步；
- internal_daily_full：包含社区集合和内部扩展模型。

这样可以避免社区标准更新后，外部 106 看板历史数据被悄悄改变口径。

### 8.2 结果字段建议

每条模型结果至少携带：

~~~~text
suite
model
mode                  # training / inference
backend               # default / dvm / triton_experimental
dynamic               # on / off
dtype
batch_size
accuracy_status
performance_status
eager_e2e_ms
compile_e2e_ms
eager_op_ms
compile_op_ms
model_wall_time_s
timeout_s
skip_reason
error_category
torch_version
torch_npu_version_or_commit
cann_version
benchmark_commit
~~~~

尤其要把 model_wall_time_s 与 E2E 平均 step time 区分开：前者回答“一个模型从开始到结束消耗多久”，后者回答“模型稳定运行阶段每 step 多快”。

### 8.3 错误状态建议

~~~~text
not_run
skipped_by_community_policy
skipped_by_npu_policy
data_missing
dependency_missing
network_download_failed
oom
timeout
compile_error
eager_error
fail_accuracy
pass_accuracy
pass_performance
~~~~

skip、fail_to_run、fail_accuracy、timeout 和 oom 不应都折叠为一个“失败”。社区 YAML 已经证明，显式记录例外是稳定看护的必要组成部分。

## 9. 推荐复用命令

### 9.1 单模型验证

~~~~bash
python benchmarks/dynamo/torchbench.py \
  --accuracy --training --amp \
  --backend=inductor --device=cuda \
  --only=<MODEL>

python benchmarks/dynamo/huggingface.py \
  --accuracy --training --amp \
  --backend=inductor --device=cuda \
  --only=<MODEL>

python benchmarks/dynamo/timm_models.py \
  --accuracy --training --amp \
  --backend=inductor --device=cuda \
  --only=<MODEL>
~~~~

### 9.2 完整 accuracy/performance

~~~~bash
# training
<runner> --accuracy --training --amp --backend=inductor
<runner> --performance --training --amp --backend=inductor

# inference
<runner> --accuracy --inference --bfloat16 --backend=inductor
<runner> --performance --inference --bfloat16 --backend=inductor
~~~~

dynamic、cudagraph、freezing、max-autotune 等属于额外 benchmark profile，需要在结果中显式记录，不能把不同 profile 合并成一条“默认性能”。

## 10. 参考来源

- [PyTorch benchmarks/dynamo README](https://github.com/pytorch/pytorch/blob/main/benchmarks/dynamo/README.md)
- [torchbench.yaml](https://github.com/pytorch/pytorch/blob/main/benchmarks/dynamo/torchbench.yaml)
- [huggingface.yaml](https://github.com/pytorch/pytorch/blob/main/benchmarks/dynamo/huggingface.yaml)
- [timm_models.yaml](https://github.com/pytorch/pytorch/blob/main/benchmarks/dynamo/timm_models.yaml)
- [huggingface_models_list.txt](https://github.com/pytorch/pytorch/blob/main/benchmarks/dynamo/huggingface_models_list.txt)
- [timm_models_list.txt](https://github.com/pytorch/pytorch/blob/main/benchmarks/dynamo/timm_models_list.txt)
- [huggingface.py](https://github.com/pytorch/pytorch/blob/main/benchmarks/dynamo/huggingface.py)
- [timm_models.py](https://github.com/pytorch/pytorch/blob/main/benchmarks/dynamo/timm_models.py)
- [inductor-perf-test-nightly.yml](https://github.com/pytorch/pytorch/blob/main/.github/workflows/inductor-perf-test-nightly.yml)
- [BenchBoard 的社区 106 模型配置](../auto_board/run_community_106.yaml)

