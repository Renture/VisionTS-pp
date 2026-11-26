# VisionTS++ 课程改进实验指南（GPU 1 分阶段执行）

本仓库基于官方 [HALF111/VisionTSpp](https://github.com/HALF111/VisionTSpp)（保留在 `upstream-base` 标签），新增 `visiontspp_course/` 独立实验套件：

- **零额外依赖**：完全适配容器现有 `visionts` 环境（Python 3.8 + PyTorch 1.7.1 + torchvision 0.8.2 + timm 0.3.2），无需安装 `uni2ts` / PyTorch 2.x。
- **双卡并行与显存保护**：默认通过 `nvidia-smi` UUID 并行调度到物理 **GPU 2 和 GPU 3**，并将单步图像上限设为 `--max-imgs-per-step 32`（峰值显存约 5~7 GB/卡，可在 24 GB 3090 上与 Table 21 稳定共存）。
- **统一结果目录**：所有阶段的日志（`logs/*.log`）、权重与单次指标（`runs/`）、总表（`runs.csv`、`summary.csv`）统一保存在 `save/course_exp/` 下。

## 1. 容器拉取与初始化

```bash
cd ~
git clone https://github.com/Renture/VisionTS-pp.git
cd VisionTS-pp
conda activate visionts
bash visiontspp_course/scripts/setup_container.sh
```

> 如需在容器拉取最新更新：`git pull --ff-only`。  
> 建议在 `tmux new -s visionts-pp` 中执行训练脚本，断线后用 `tmux attach -t visionts-pp` 接回。

## 2. 快速冒烟验证（约 30 秒，强烈建议先跑）

验证 Step 1（彩色多变量合图 + 窗口相关性分组）与 Step 2（多分位数头 + Soft Clipping + LoRA）全链路无报错：

```bash
bash visiontspp_course/scripts/run_step1_smoke.sh
```

## 3. 第一步：多变量彩色合图 + 动态相关性分组（MSE + 仅微调 LN）

### 3.1 快速验收（ETTm1 全 4 步长 + Weather 96/192，同时对比 `half` 与 `align_const`）

```bash
bash visiontspp_course/scripts/run_step1_quick.sh
```

结果输出至 `save/course_exp/step1_quick/summary.csv`。

### 3.2 分组与颜色消融（`all` vs `sequential` vs `train_corr` vs `window_corr`，`gray` vs `random_adj`）

根据 3.1 选出较优对齐方式（`half` 或 `align_const`）后运行：

```bash
ALIGN_MODE=half bash visiontspp_course/scripts/run_step1_ablation.sh
```

## 4. 第二步：多分位数概率头 + 轻量边界适配 + 微调范围调优

### 4.1 多分位数头（`q5` / `q9`）+ Pinball Loss + CRPS

```bash
ALIGN_MODE=half bash visiontspp_course/scripts/run_step2_quantile.sh
```

### 4.2 边界适配对照（`none` vs `hard_clip` vs `filter_train` vs `soft_clip`，ETTm2 & Weather）

```bash
ALIGN_MODE=half QUANTILE_MODE=q5 bash visiontspp_course/scripts/run_step2_boundary.sh
```

### 4.3 微调范围对照（`ln_head` vs `ln_head_patch` vs `ln_head_lora`，ETTm1，1~3 epoch 早停）

```bash
ALIGN_MODE=half QUANTILE_MODE=q5 CLIP_MODE=soft_clip bash visiontspp_course/scripts/run_step2_ft_scope.sh
```

## 5. 自定义单次实验示例

```bash
python -u visiontspp_course/run_experiment.py \
  --stage custom --gpu 1 \
  --datasets Weather --horizons 96 192 --seeds 2021 2022 2023 \
  --grouping window_corr --max-group-size 7 --sim-metric pearson \
  --color-mode random_adj --align-mode half \
  --quantile-mode q5 --clip-mode soft_clip --ft-type ln_head_lora \
  --epochs 3 --patience 2
```
