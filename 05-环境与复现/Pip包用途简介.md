# Pip 包用途简介

本文解释 `practical_migration_guide.md` 中出现的 pip 包用途。重复出现在多个层里的包只说明一次。

说明：

- “未确认”表示没有找到可靠公开用途，或包名过于通用，建议以实际安装源和代码 import 为准。
- 这里是迁移理解用的简介，不代表所有包都是运行时强依赖。

## 公共 Runtime 包

- `certifi`：提供 Mozilla CA 根证书集合，供 HTTPS 请求验证证书使用。
- `charset-normalizer`：文本编码检测库，常作为 `requests` 的依赖。
- `cloudpickle`：增强版 pickle，支持序列化普通 pickle 不支持的 Python 对象，常用于分布式/并行任务。
- `exceptiongroup`：为旧 Python 版本提供 `ExceptionGroup` 支持，常见于异步和测试生态。
- `farama-notifications`：Farama/Gymnasium 生态的通知信息包。
- `filelock`：跨平台文件锁，用于缓存、下载、并发写文件时避免冲突。
- `fsspec`：统一文件系统接口，支持本地、对象存储、远端文件系统等。
- `gymnasium`：强化学习环境接口库，是 OpenAI Gym 的后续维护版本。
- `hf-xet`：Hugging Face Hub 使用的 Xet 存储/下载加速相关组件。
- `huggingface-hub`：访问 Hugging Face Hub 的客户端库，用于下载模型、数据集和文件。
- `idna`：国际化域名编码支持，常作为 HTTP 网络库依赖。
- `jinja2`：模板引擎，常用于生成文本、HTML、配置或模型卡片。
- `markupsafe`：为 Jinja2 等模板库提供安全字符串转义。
- `mfusion`：未确认。没有搜到可靠公开 PyPI 用途，可能是内部包或项目私有依赖。
- `ml-dtypes`：机器学习专用 dtype 扩展，例如 bfloat16、float8 等。
- `mpmath`：纯 Python 高精度数学库，常被 `sympy` 依赖。
- `networkx`：图结构和图算法库，PyTorch/模型转换工具中也可能间接使用。
- `numpy`：基础数组和数值计算库。
- `pandas`：表格数据处理和分析库。
- `pillow`：图像读写和处理库。
- `pybind11`：C++/Python 绑定工具，常用于构建 Python C++ 扩展。
- `python-dateutil`：日期时间解析和时区/周期规则处理。
- `pytz`：时区数据库和时区转换支持。
- `pyyaml`：YAML 文件读写。
- `regex`：增强版正则表达式库，功能比 Python 标准库 `re` 更强。
- `requests`：常用 HTTP 客户端库。
- `six`：Python 2/3 兼容辅助库，很多旧依赖仍会用到。
- `sympy`：符号数学库，PyTorch 形状推导/编译相关路径中常见。
- `tiktoken`：OpenAI tokenizer 库，用于 token 计数和编码。
- `torch-mlir`：PyTorch 到 MLIR 的编译/转换相关组件。
- `triton-ascend`：Ascend NPU 方向的 Triton 相关后端/适配组件。
- `typing-extensions`：为旧 Python 版本补充新的类型注解能力。
- `tzdata`：时区数据包。
- `urllib3`：底层 HTTP 客户端库，常被 `requests` 使用。

## NPU 版本栈包

- `torch`：PyTorch 主框架，提供张量计算、自动求导、模型定义和运行能力。
- `torch-npu`：PyTorch 的 Ascend NPU 适配插件，让 PyTorch 能在 NPU 上运行。
- `torchvision`：PyTorch 视觉库，提供图像模型、图像变换和视觉数据集工具。

## Bench Runtime 包

- `ast-serialize`：mypy/mypyc 生态的 Python AST 解析和序列化扩展。
- `attrs`：简化 Python 类定义的数据类工具库，很多库用它定义配置/结构化对象。
- `blobfile`：统一访问本地文件、云存储对象等 blob 文件的工具库。
- `dall-e`：OpenAI DALL-E 相关旧包，常见于图像生成/多模态示例依赖。
- `decorator`：帮助编写行为良好的 Python 装饰器。
- `ftfy`：修复乱码、错误 Unicode 文本和异常编码问题。
- `gym-notices`：旧 Gym 生态的通知/迁移提示包。
- `iopath`：Meta/Facebook 的 I/O 抽象库，用于统一本地和远端路径访问。
- `kaldi-io`：读取 Kaldi 语音工具链数据格式的 Python 库。
- `librt`：mypyc runtime library，提供 mypyc 编译代码可用的运行时支持和优化实现。
- `lxml`：高性能 XML/HTML 解析和处理库。
- `opencv-python-headless`：无 GUI 依赖的 OpenCV Python 包，用于图像/视频处理。
- `portalocker`：跨平台文件锁库，常用于多进程环境下保护文件访问。
- `protobuf`：Google Protocol Buffers 序列化库，很多模型和服务接口会用到。
- `pycryptodomex`：加密算法库，是 PyCryptodome 的独立命名空间版本。
- `psutil`：读取系统进程、CPU、内存、磁盘等运行状态。
- `safetensors`：安全、高效的张量权重文件格式读写库。
- `scipy`：科学计算库，提供线性代数、优化、统计、信号处理等功能。
- `tensorboardx`：TensorBoard 日志写入库，可在非 TensorFlow 项目中记录训练/性能指标。
- `timm`：PyTorch Image Models，提供大量视觉模型和预训练权重接口。
- `tokenizers`：Hugging Face 高性能 tokenizer 实现。
- `torchmultimodal`：PyTorch 多模态模型相关库。
- `transformers`：Hugging Face Transformers，提供 Transformer 模型、配置、tokenizer 和推理/训练工具。

## LLM Runtime 包

- `accelerate`：Hugging Face 的训练/推理加速和多设备分发辅助库。
- `aiohappyeyeballs`：异步网络连接优化组件，实现 Happy Eyeballs 连接策略。
- `aiohttp`：异步 HTTP 客户端/服务端库。
- `aiosignal`：aiohttp 生态使用的异步信号/回调管理库。
- `anyio`：统一 asyncio、trio 等异步后端的抽象层。
- `async-timeout`：异步操作超时控制工具。
- `datasets`：Hugging Face Datasets，用于加载、处理和缓存数据集。
- `diffusers`：Hugging Face Diffusers，主要用于扩散模型和生成式模型 pipeline。
- `dill`：增强版序列化库，能序列化更多 Python 对象，常用于多进程/数据集处理。
- `frozenlist`：不可变列表结构，aiohttp/yarl 生态依赖。
- `h11`：纯 Python HTTP/1.1 协议实现。
- `httpcore`：HTTPX 的底层 HTTP 传输核心。
- `httpx`：现代 HTTP 客户端，支持同步和异步请求。
- `importlib-metadata`：读取 Python 包元数据，旧 Python 版本的兼容实现。
- `multidict`：支持一个 key 对应多个 value 的字典结构，aiohttp 生态依赖。
- `multiprocess`：增强版 multiprocessing，常和 `dill` 配合用于数据集并行处理。
- `peft`：Hugging Face PEFT，用于 LoRA 等参数高效微调。
- `propcache`：属性/计算结果缓存工具，常作为 yarl/aiohttp 依赖。
- `pyarrow`：Apache Arrow Python 库，用于高效列式数据、数据集缓存和 parquet 等格式。
- `sentencepiece`：Google SentencePiece tokenizer，用于很多 LLM 的分词。
- `tqdm`：进度条显示库。
- `xxhash`：高速非加密哈希库，常用于数据集缓存和去重。
- `yarl`：URL 构造和解析库，aiohttp 生态依赖。
- `zipp`：zip 文件路径访问工具，常作为 importlib 相关依赖。
