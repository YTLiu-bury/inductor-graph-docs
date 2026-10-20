# 编译 torch_npu whl

## 方案说明

**一个公用镜像 + 三个常驻容器**：

| 步骤 | 脚本 | 作用 |
|------|------|------|
| 1 | `remote_env/compile_whl/docker/build_compile_images.sh` | 确认/构建公用镜像 `manylinux-builder:v1` |
| 2 | `remote_env/compile_whl/docker/start_compile_containers.sh` | 用该镜像起 3 个容器，各预装对应 `torch_cpu` |
| 3 | `git_commit/git_compile.py` | `docker exec` 进对应容器编译 |

| PTA 版本 | 容器名 | 容器内 torch |
|----------|--------|--------------|
| v2.7.1 | `bb-compile-v271` | `2.7.1+cpu` |
| v2.9.0 | `bb-compile-v290` | `2.9.0+cpu` |
| v2.10.0 | `bb-compile-v210` | `2.10.0+cpu` |

## 标准流程（按顺序执行）

```bash
cd <BenchBoard>

# .sh 文件通过 .gitattributes 固定为 LF，可直接在 Linux 服务器执行。

# Step 1：检查公用镜像（你机器上已有 v1 时会直接通过）
bash remote_env/compile_whl/docker/build_compile_images.sh

# Step 2：启动三个编包容器并初始化 vim-common / torch_cpu
bash remote_env/compile_whl/docker/start_compile_containers.sh

# Step 3：编译
python git_commit/git_compile.py --date 20260521 --force
```

若本机还没有 `manylinux-builder:v1`，在 Ascend pytorch 源码目录构建后再跑 Step 1：

```bash
cd <pytorch>/ci/docker/X86
docker build -t manylinux-builder:v1 .

# 或指定目录让 Step 1 自动构建
PYTORCH_DOCKER_DIR=<pytorch>/ci/docker/X86 bash remote_env/compile_whl/docker/build_compile_images.sh
```

## 重建容器环境

```bash
bash remote_env/compile_whl/docker/stop_compile_containers.sh
bash remote_env/compile_whl/docker/start_compile_containers.sh
```

## 每日远程任务（只跑 commit + 编译）

`remote_whl_tasks.sh` **不会**每天执行 build/start（那两步是一次性准备）。
首次或容器被删后，需先手动完成 Step 1、Step 2，再跑每日任务：

```bash
# 一次性（或容器丢失后重做）
bash remote_env/compile_whl/docker/build_compile_images.sh
bash remote_env/compile_whl/docker/start_compile_containers.sh

# 每天
bash git_commit/remote_whl_tasks.sh 20260521
```

`build` / `start` 均有防呆：镜像已存在、容器已在跑、`.bb_compile_ready_*` 标记存在时会跳过，不会重复构建/重装。
