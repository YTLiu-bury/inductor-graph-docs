# Inductor Graph Docs

PyTorch Inductor、NPU 图模式、Triton Experimental 与 Benchmark 看护知识库。

本仓用于沉淀可复用的技术分析、问题定位方法、流水架构和上游对齐经验。文档以中文文件名维护，便于团队检索；仓库根目录保留 `README.md` 作为 GitHub 默认入口。

## 文档目录

| 目录 | 内容 |
| --- | --- |
| [01-图模式与编译器](01-图模式与编译器/) | Inductor、Triton Experimental、图模式、FlexAttention、RNG、Profiler 和 NPU patch |
| [02-Benchmark流水](02-Benchmark流水/) | 模型看护、流水入口、版本化运行、社区模型和看板数据链路 |
| [03-社区对齐](03-社区对齐/) | PyTorch 社区 YAML、PTA 看护逻辑、A100 复现和对齐方案 |
| [04-融合与性能](04-融合与性能/) | 融合范围 GAP、Kernel 性能、E2E 与 OP 数据分析方法 |
| [05-环境与复现](05-环境与复现/) | 环境迁移、依赖、Benchmark 仓库和 torch_npu 编译复现 |
| [06-案例与PR](06-案例与PR/) | PR 说明和可复用的案例材料 |

## 维护原则

1. 技术结论尽量附源码、日志字段或上游链接，区分“已验证”“推断”和“待验证”。
2. Benchmark 运行文档记录入口、参数、产物契约和数据发布链路，不提交 rawdata、profile、模型权重、缓存和网站 JSON。
3. 服务器路径、IP、用户名、密码、Token、代理地址和内部凭据统一使用占位符，不进入仓库。
4. 上游行为与 NPU 特化分开描述，避免把 NPU 流水策略误写成 PyTorch 社区通用规则。
5. 新增文档先放入对应分类；暂时无法归类的内容放入待整理区，不直接混入主目录。

## 文档状态

当前内容由既有 Benchmark、Inductor 和 NPU 问题记录整理而来。原始工作仓中的文件不被本仓反向修改；本仓副本会持续做脱敏、校正和结构化维护。

## 相关代码仓

- [BenchmarkForge](https://github.com/YTLiu-bury/BenchmarkForge)：Benchmark 流水架构与运行工具
- [PyTorch](https://github.com/pytorch/pytorch)：Inductor 与社区 Benchmark runner
- [Ascend/pytorch](https://github.com/Ascend/pytorch)：PyTorch NPU 适配和 NPU Inductor 后端
- [Triton](https://github.com/triton-lang/triton)：GPU Triton 基线实现
- [Triton Ascend](https://github.com/triton-lang/triton-ascend)：Ascend Triton 实现

