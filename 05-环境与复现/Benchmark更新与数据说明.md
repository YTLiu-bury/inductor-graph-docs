1. 准备 benchmarks 下的 torchbench 和 llm 目录

v2.7.1 分支的 benchmarks 目录已经包含 torchbench 和 llm，不再需要从私仓捕获 llm。

```bash
git clone https://gitcode.com/Ascend/pytorch.git
cd pytorch
git checkout v2.7.1
cd ..

mkdir -p benchmarks
cp -r pytorch/benchmarks/torchbench benchmarks/
cp -r pytorch/benchmarks/llm benchmarks/
```

2. 准备 benchmarks/torchbench 下的 benchmark 目录

benchmarks/torchbench/benchmark 仍需要从 PyTorch benchmark 仓库 clone，并固定到已验证 commit。

建议默认使用较新的官方 benchmark commit：

- `fb14629994956c16d27c7a9940c5f081afbf68dc`

这个版本比原来的 `9910b31cc17d175a781412fd9ca6f18a4ee04610` 新很多，已经在 `PR 45810` 的 torchbench/common/npu_support 适配基础上做过 smoke 验证。原 `9910b31cc17d175a781412fd9ca6f18a4ee04610` 只作为保守回退点保留。

```bash
cd benchmarks/torchbench
git clone https://github.com/pytorch/benchmark.git --depth=1
cd benchmark
git remote set-branches origin main
git fetch --depth 1 origin fb14629994956c16d27c7a9940c5f081afbf68dc
git checkout fb14629994956c16d27c7a9940c5f081afbf68dc
cd ../../..
```

如果新版本出现未覆盖的问题，可以临时回退到旧 pin：

```bash
cd benchmarks/torchbench/benchmark
git fetch --depth 1 origin 9910b31cc17d175a781412fd9ca6f18a4ee04610
git checkout 9910b31cc17d175a781412fd9ca6f18a4ee04610
```

3. 新 benchmark commit 的适配点

官方较新的 `pytorch/benchmark` 已经把 `benchmark/userbenchmark/dynamo/dynamobench/torchbench.py` 里的部分常量迁移到 `torchbench.yaml`，Ascend 侧 `benchmarks/torchbench/torchbench.py` 不能再只依赖旧常量导入。需要在 Ascend wrapper 里做兼容：

- 旧 benchmark commit：继续从 `benchmark.userbenchmark.dynamo.dynamobench.torchbench` 导入 `DONT_CHANGE_BATCH_SIZE` 等常量。
- 新 benchmark commit：如果导入失败，则通过 `benchmark.userbenchmark.dynamo.dynamobench.common.load_yaml_file("torchbench.yaml")` 读取 `dont_change_batch_size`、`only_training`、`batch_size.training`、`accuracy.max_batch_size`、`dtype.force_amp_for_fp16_bf16_models`、`tolerance.higher`、`tolerance.even_higher`、`slow`、`very_slow`、`non_deterministic`。

另外，较新的 detectron2 数据加载逻辑仍然会检查仓内路径 `benchmark/torchbenchmark/data/.data/coco2017-minimal`，不会只依赖 `TORCHBENCH_DATA_PATH`。如果数据已经统一放在 `<WORKSPACE_ROOT>/benchmark/torchbenchmark/data/.data`，需要补一个软链：

```bash
cd benchmarks/torchbench
mkdir -p benchmark/torchbenchmark/data/.data
ln -sfn <WORKSPACE_ROOT>/benchmark/torchbenchmark/data/.data/coco2017-minimal \
  benchmark/torchbenchmark/data/.data/coco2017-minimal
```

`PR 45810` 后 `--disable-aclgraph` 会同时关闭 NPU aclgraph 和 Inductor Triton cudagraph/cudagraph_trees。对比新旧性能数据时需要记录是否带 `--disable-aclgraph`，避免把不同执行条件的数据直接混比。

4. 清理临时目录

```bash
rm -rf pytorch
```
