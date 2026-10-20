# 2.13 / 2.15 版本化 Benchmark 流水

## 目标

每天启动 daily 或 weekly 前，分别为 PyTorch 2.13 和 2.15 编译、安装并校验 `torch_npu`，随后在对应环境中运行 DVM 和 Triton-Experimental。两套环境、编译目录、wheel 目录和产物版本标签彼此隔离。

## 服务器配置

```bash
cp config/versioned_benchmark_pipeline.example.env \
  <WORKSPACE_ROOT>/ci/versioned_benchmark_pipeline.env
```

修改服务器上的配置，至少确认：

- `CONDA_ENV_213` / `CONDA_ENV_215`；
- 如果环境是 `conda-pack`，配置 `ENV_ACTIVATE_213` / `ENV_ACTIVATE_215`，不要假设存在 `conda.sh`；
- 两个构建环境都需要 `cmake`、`ninja`（若构建目标启用）和 `nanobind`；
- 两个版本各自的源码目录、编译工作目录和 `BUILD_COMMAND`；
- 两个版本各自的 wheel 目录和文件匹配规则；
- DVM/Triton 使用的 NPU 卡不重叠。

仓库不保存服务器上的实际配置，也不保存 token、代理和密码。

## 一键启动

在 BenchBoard 根目录执行：

```bash
bash scripts/run_versioned_benchmark_pipeline.sh \
  --daily --date 20261019
```

执行顺序是：

1. 在临时、干净 worktree 中编译 2.13；
2. 安装最新的 2.13 `torch_npu` wheel 并校验 import；
3. 在临时、干净 worktree 中编译 2.15；
4. 安装最新的 2.15 `torch_npu` wheel 并校验 import；
5. 两套环境都校验成功后，分别运行 daily 的 Bench、HuggingFace、Timm；
6. 每个版本下 DVM 与 Triton-Experimental 使用不同的设备集合并行，单个后端内的 suite 串行。

## 卡数口径

当前 `run_model.py` 对一个 suite 内的模型是串行执行的。`ASCEND_RT_VISIBLE_DEVICES=0,1`
只表示进程可以看到两张卡，不代表一个 suite 会并行使用两张卡；因此当前默认调度是每个版本两个
backend worker，不能据此宣称 8 张卡都在跑。若要稳定利用 8 张卡，需要额外把模型列表拆成多个
worker，并让每个 worker 使用单独的 NPU；这属于调度改造，不能只把环境变量改成逗号分隔列表。

weekly：

```bash
bash scripts/run_versioned_benchmark_pipeline.sh \
  --weekly --date 20261019
```

如需把 LLM suite 也加入 weekly：

```bash
bash scripts/run_versioned_benchmark_pipeline.sh \
  --weekly --include-llm --date 20261019
```

只编包、安装和校验，不启动模型：

```bash
bash scripts/run_versioned_benchmark_pipeline.sh \
  --build-only --date 20261019
```

## 安全边界

- 编包失败或 `torch/torch_npu` 校验失败时，不会启动 Benchmark；
- 2.13 和 2.15 使用独立 Conda 环境，不覆盖另一套环境；
- 编包脚本使用临时 worktree，构建结束自动清理，不修改主源码树；
- 通过 `flock` 防止两个版本化任务同时编包或启动；
- 旧的 `auto_board/remote_tasks.sh` 是历史入口，不建议用它替代此脚本；
- 运行日志位于 `logs/versioned_pipeline/<date>/<version>/`，模型 raw data 仍由 `auto_board_run.sh` 按版本标签写入 `board_file`。

## 重要说明

当前脚本只负责调度，不能凭空推断 2.15 的编包命令。必须在服务器配置中填写已经验证过的 2.15 编包入口；如果 2.15 仍只有手工命令，就先把手工命令整理成 `BUILD_COMMAND_215`，再接入一键入口。
