# 轻量 Profiler 环境变量控制

## 背景

Benchmark 流水需要同时采集 E2E 耗时和算子耗时。原有 NPU Profiler 默认在 warmup 后持续采集剩余 step，模型迭代次数较多时会产生较大的 profiler 产物，并延长数据处理时间。

## 改动位置

- `third_party/benchmarks/torchbench/common.py`
  - 读取 `BENCH_PROFILE_WARMUP` 和 `BENCH_PROFILE_ACTIVE`。
  - 将计算出的采集窗口传给 `NPUProfiler`。
- `third_party/benchmarks/torchbench/profiler.py`
  - 使用传入的 `warmup` 和 `active` 构造 NPU Profiler schedule。
  - 每个模型 step 通过 `prof.step()` 推进 schedule。

## 采集逻辑

NPU Profiler 使用以下 schedule：

```python
torch_npu.profiler.schedule(
    wait=0,
    warmup=warmup,
    active=active,
    repeat=1,
)
```

其中：

- `warmup`：预热 step 数，不作为正式 profiler 数据保存。
- `active`：正式采集 step 数。
- `repeat=1`：只采集一次，完成后停止继续采集。

模型本身仍然会按照原有配置跑完整 iteration；这两个变量只控制 profiler 采集窗口，不会改变模型的训练、推理、编译或精度校验逻辑。

## 环境变量

### `BENCH_PROFILE_WARMUP`

控制 warmup step 数，默认值为 `10`。非法值会回退到 `10`，最小值为 `0`。

### `BENCH_PROFILE_ACTIVE`

控制正式采集 step 数，默认值为 `n - warmup`，其中 `n` 是当前模型的 iteration 数。非法值会回退到剩余 step 数，最小值为 `1`。

当设置值超过剩余 iteration 数时，会自动限制在剩余 step 数以内；当 warmup 不小于总 iteration 数时，会自动调整为至少留出一个可采集 step。

## 使用示例

只采集一个正式 step：

```bash
BENCH_PROFILE_WARMUP=2 \
BENCH_PROFILE_ACTIVE=1 \
bash daily_pipeline.sh
```

也可以在当前 shell 中设置后运行多个任务：

```bash
export BENCH_PROFILE_WARMUP=2
export BENCH_PROFILE_ACTIVE=1
bash daily_pipeline.sh
```

此配置表示前两个 step 用于预热，第三个 step 用于正式 profiler 采集，后续 step 不再写入 profiler trace。

## 默认行为与兼容性

不设置环境变量时，保持默认行为：

```text
BENCH_PROFILE_WARMUP=10
BENCH_PROFILE_ACTIVE=n-10
```

因此已有流水命令不需要修改即可运行。CUDA Profiler 的配置不受这两个环境变量影响，目前仍使用其原有固定配置。

## 数据可信度说明

轻量采集可以明显减少产物大小和 profiler 开销，但正式采集 step 越少，算子平均耗时的统计波动越大：

- 日常全量流水：建议使用 `active=1` 或 `active=2`，主要用于获得算子耗时趋势。
- 性能回归分析：建议使用 `active=3` 到 `active=5`，并结合多个重复运行结果判断。
- 精细算子性能研究：不建议只依赖单个 active step，应增加采集次数并固定环境、输入和 iteration 配置。

该改动不会改变 E2E 耗时统计。E2E 仍由完整模型运行过程计算，OP 数据则来自 profiler 的 active 窗口。
