<!--
PR描述模板更新日期：20260203
-->

# 【合入来源】
> <font color="red">**如有社区issue，请关联issue链接**</font>\
> <font color="red">**请勿携带内部流程信息（需求链接、问题单、内部issue等）**</font>

- [ ] 需求
- [ ] 问题单
- [ ] issue/工单
- [ ] 重构优化
- [x] 资料更新

# 【修改方案】
1. 更新 `benchmarks/torchbench/README.md` 中 `pytorch/benchmark` 的推荐依赖版本，将默认 checkout commit 从 `9910b31cc17d175a781412fd9ca6f18a4ee04610` 更新为 `fb14629994956c16d27c7a9940c5f081afbf68dc`，并保留旧 commit 作为临时回退方式。
2. 在 `benchmarks/torchbench/torchbench.py` 中增加新版 `pytorch/benchmark` 元数据兼容逻辑。旧版本仍从 `benchmark.userbenchmark.dynamo.dynamobench.torchbench` 导入常量；新版不再导出这些常量时，改为从 `torchbench.yaml` 读取等价配置。
3. 该修改仅适配 `pytorch/benchmark` 配置结构变化，不改变 TorchBench runner 执行流程、命令行参数和 NPU backend 选择逻辑。

# 【资料变更】
涉及资料变更。

更新 TorchBench 使用文档中的 `pytorch/benchmark` 推荐 commit，并补充旧 commit 的回退命令，便于在新版 benchmark 出现兼容问题时快速恢复到历史验证版本。

# 【接口变更】
不涉及客户面可见接口变更。

本次修改不新增、不删除命令行参数，不改变 `--backend`、`--npu-backend`、`--training`、`--inference` 等已有使用方式。

# 【功能验证】
已完成以下验证：

```bash
python -m py_compile benchmarks/torchbench/torchbench.py
```

基于新版 `pytorch/benchmark` commit `fb14629994956c16d27c7a9940c5f081afbf68dc` 执行：

```bash
python torchbench.py --help
```

执行 NPU smoke 验证：

```bash
python torchbench.py --devices npu --backend inductor \
  --npu-backend triton_experimental --disable-aclgraph \
  --accuracy --training --only alexnet --iterations 1 --timeout 900
```

验证结果：

```text
npu train alexnet
pass_accuracy
exit code: 0
```

# 【CheckList】
> PR提交人对以下CheckList自检项进行全量自检，自检通过或不涉及，均修改 [ ] 为 [x]

- [x] 代码注释完备，正确记录错误日志
- [x] 代码实现进行了返回值、空指针等校验
- [x] PR标题正确使用类型标签，如：feat、fix、refactor、docs、test等
- [ ] PR持续集成流水线（CI）执行通过，代码检查无异常
