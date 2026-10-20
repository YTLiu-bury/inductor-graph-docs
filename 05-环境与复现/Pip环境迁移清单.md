# Pip 环境迁移清单

目标环境：

- `bench_v271`
- `bench_v290`
- `bench_v2100`
- `llm_v271`
- `llm_v290`
- `llm_v2100`

安装脚本 `create_six_conda_envs.sh` 默认显式使用 PyPI 官方源，并在普通包安装失败时尝试 fallback 源，避免机器上的 pip 全局镜像配置导致 403 或下载异常。可按服务器网络情况覆盖：

```bash
PYPI_INDEX_URL=https://pypi.org/simple \
PYPI_FALLBACK_INDEX_URL=https://mirrors.aliyun.com/pypi/simple \
PYTORCH_CPU_INDEX=https://download.pytorch.org/whl/cpu \
bash create_six_conda_envs.sh
```

本文只整理 pip 迁移需要关注的包。conda 导出里的 gcc、openssl、sqlite、zlib 等底层包由目标服务器新建 conda 环境时自动解析，不在这里展开。

## 1. 公共 Pip Runtime

所有 6 个环境共用：

```text
certifi==2026.4.22
charset-normalizer==3.4.7
cloudpickle==3.1.2
decorator==5.1.1
exceptiongroup==1.3.1
farama-notifications==0.0.6
filelock==3.29.4
fsspec==2026.6.0
gymnasium==1.3.0
hf-xet==1.5.0
huggingface-hub==0.36.2
idna==3.15
jinja2==3.1.6
markupsafe==3.0.3
mfusion==1.0
ml-dtypes==0.5.4
mpmath==1.3.0
networkx==3.4.2
numpy==2.2.6
pandas==2.3.3
pillow==12.2.0
pybind11==3.0.4
python-dateutil==2.9.0.post0
pytz==2026.2
pyyaml==6.0.3
regex==2026.5.9
requests==2.34.2
scipy==1.13.1
six==1.17.0
sympy==1.14.0
tensorboardx==2.6.5
tiktoken: <REDACTED>
torch-mlir==0.0.1
triton-ascend==3.2.0
typing-extensions==4.15.0
tzdata==2026.2
urllib3==2.7.0
```

## 2. NPU 版本栈

`triton-ascend==3.2.0` 已放在公共 runtime 中。每个版本只需要按目标环境选择对应的 `torch`、`torch-npu`、`torchvision`：

```text
v271:
torch==2.7.1
torch-npu==2.7.1.post5+git33d567f
torchvision==0.22.1

v290:
torch==2.9.0
torch-npu==2.9.0.post3+gitee63b0a
torchvision==0.24.0

v2100:
torch==2.10.0
torch-npu==2.10.0.post1+gitabde8b2
torchvision==0.25.0
```

## 3. Bench/LLM 同名不同版本 Runtime

这些包两类环境都会安装，但版本不同。

建议保留 workload 差异的包：

```text
bench:
protobuf==7.34.1
safetensors==0.7.0
tokenizers==0.15.2
transformers==4.36.0

llm:
protobuf==6.33.5
safetensors==0.6.2
tokenizers==0.22.2
transformers==4.57.1
```

`safetensors`、`tokenizers`、`transformers` 属于 Hugging Face 模型栈，版本之间确实有配套关系，建议按 `bench`/`llm` 分开。`protobuf` 也建议保守跟随 workload，因为它经常被模型、数据和序列化相关库间接使用。

差异不太关键、可以考虑抹平的包：

```text
bench:
attrs==24.2.0
psutil==6.0.0

llm:
attrs==26.1.0
psutil==7.2.2
```

`attrs` 是通用数据类/结构化对象工具，`psutil` 是系统资源读取工具，通常不和 torch 或 Hugging Face 版本强绑定。迁移时如果想更简单，可以统一成一套版本；如果想完全贴近导出环境，就继续按上面分开。

## 4. Bench 独有 Runtime

迁移 `bench_*` 环境时，在公共 runtime、NPU 版本栈和 bench 版本包之外，添加：

```text
ast-serialize==0.4.0
blobfile==3.2.0
dall-e==0.1
ftfy==6.3.1
gym-notices==0.1.0
iopath==0.1.10
kaldi-io==0.9.8
librt==0.11.0
lxml==6.1.0
opencv-python-headless==4.13.0.92
portalocker==3.2.0
pycryptodomex==3.23.0
timm==1.0.27
torchmultimodal==0.1.0b0
```

## 5. LLM 独有 Runtime

迁移 `llm_*` 环境时，在公共 runtime、NPU 版本栈和 llm 版本包之外，添加：

```text
accelerate==1.11.0
aiohappyeyeballs==2.6.1
aiohttp==3.13.5
aiosignal==1.4.0
anyio==4.13.0
async-timeout==5.0.1
datasets==4.5.0
diffusers==0.37.0
dill==0.4.0
frozenlist==1.8.0
h11==0.16.0
httpcore==1.0.9
httpx==0.28.1
importlib-metadata==9.0.0
multidict==6.7.1
multiprocess==0.70.18
peft==0.17.1
propcache==0.5.2
pyarrow==23.0.0
sentencepiece==0.2.1
tqdm==4.67.1
xxhash==3.7.0
yarl==1.23.0
zipp==3.23.1
```

## 6. 组合方式

```text
bench_v271 = 公共 Pip Runtime + v271 NPU 版本栈 + bench 同名不同版本 Runtime + Bench 独有 Runtime
bench_v290 = 公共 Pip Runtime + v290 NPU 版本栈 + bench 同名不同版本 Runtime + Bench 独有 Runtime
bench_v2100 = 公共 Pip Runtime + v2100 NPU 版本栈 + bench 同名不同版本 Runtime + Bench 独有 Runtime

llm_v271 = 公共 Pip Runtime + v271 NPU 版本栈 + llm 同名不同版本 Runtime + LLM 独有 Runtime
llm_v290 = 公共 Pip Runtime + v290 NPU 版本栈 + llm 同名不同版本 Runtime + LLM 独有 Runtime
llm_v2100 = 公共 Pip Runtime + v2100 NPU 版本栈 + llm 同名不同版本 Runtime + LLM 独有 Runtime
```
