# AIsecurity — 对抗攻击白盒实验（MNIST + LeNet）

本项目实现并对比多种经典**白盒对抗攻击算法**，在 MNIST 手写数字识别数据集与预训练 LeNet 模型上，验证"迭代式对抗攻击相比单步攻击能显著提升白盒攻击成功率"这一结论。

## 项目简介

- **任务**：白盒攻击（攻击者完全掌握模型结构与参数），构造对抗样本使模型误分类。
- **模型**：LeNet-5 变体（卷积 32→64，全连接 128，输入 1×28×28），权重文件 `lenet_mnist_model.pth`。
- **数据集**：MNIST，固定取测试集前 N 张（默认 1000 张），仅在模型原本分类正确的样本上统计攻击成功率。
- **对比算法**：FGSM（单步基线）、PGD、MI-FGSM、C&W、DeepFool；另有 Patch 补丁攻击的参考实现。

## 目录结构

```
AIsecurity/
├── compare_attacks.py          # 统一对比实验脚本（推荐入口，可直接运行）
├── lenet_mnist_model.pth       # 预训练 LeNet 模型权重
├── fgsm.py                     # FGSM 单步攻击（参考实现）
├── pgd_attack.py               # PGD 投影梯度下降（参考实现）
├── mi-fgsm.py                  # MI-FGSM 动量迭代（参考实现）
├── cw.py                       # C&W 优化式攻击（参考实现）
├── deepfool.py                 # DeepFool 最小扰动攻击（参考实现）
├── patch_attack.py             # Patch 补丁攻击（参考实现）
├── data/                       # MNIST 数据集（首次运行自动下载）
└── outputs/                    # 实验结果输出
    ├── compare_summary.csv              # 汇总表（ASR / 扰动 / 耗时）
    ├── asr_vs_epsilon_compare.png       # ASR 随 ε 变化曲线
    ├── accuracy_vs_epsilon_compare.png  # 攻击后准确率曲线
    ├── perturbation_compare.png         # 平均扰动强度对比
    └── adv_examples_compare.png         # 对抗样本可视化
```

## 环境依赖

| 依赖 | 版本 |
|------|------|
| Python | 3.10 |
| PyTorch | 2.5.1（CUDA 12.4） |
| torchvision | 0.20.1 |
| matplotlib | ≥3.5 |
| NumPy | 1.26 |

> 本机（共享 A100 服务器）可直接使用已配置好的 conda 环境：
> ```bash
> /home/wuchenxi/.conda/envs/mixkv/bin/python
> ```

## 快速开始

```bash
cd /home/wuchenxi/projects/AIsecurity

# 快速冒烟测试（30 个样本、2 个扰动预算）
/home/wuchenxi/.conda/envs/mixkv/bin/python compare_attacks.py --samples 30 --eps 0.1,0.2

# 完整实验（1000 个样本，默认 ε=0.05,0.1,0.15,0.2,0.25,0.3）
/home/wuchenxi/.conda/envs/mixkv/bin/python compare_attacks.py --samples 1000
```

首次运行会自动下载 MNIST 到 `data/` 目录。实验结束后结果写入 `outputs/`。

### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--samples` | 参与攻击的测试样本数（只统计原本分类正确的样本） | 500 |
| `--eps` | 逗号分隔的扰动预算列表 | `0.05,0.1,0.15,0.2,0.25,0.3` |
| `--steps` | PGD / MI-FGSM 迭代步数 | 10 |
| `--cw-steps` | C&W 优化迭代步数 | 40 |
| `--deepfool-iter` | DeepFool 最大迭代次数 | 30 |
| `--model` | 模型权重路径 | `lenet_mnist_model.pth` |
| `--outdir` | 输出目录 | `outputs` |

## 攻击算法

统一接口为 `attack(model, x_denorm, y, eps) -> x_adv`，输入输出均在像素空间 `[0,1]`：

| 算法 | 类型 | 核心思想 |
|------|------|----------|
| **FGSM** | 单步基线 | 沿梯度符号方向一次性移动 ε |
| **PGD** | 迭代投影 | FGSM 迭代版：随机初始化 + 每步投影回 ℓ∞ 球内 |
| **MI-FGSM** | 动量迭代 | 在 PGD 基础上累积动量，稳定梯度方向 |
| **C&W** | 优化式 | 最小化 ‖δ‖₂² + c·f(x+δ)，生成最小 ℓ₂ 扰动 |
| **DeepFool** | 最小扰动 | 迭代逼近最近决策边界，寻找最小扰动 |
| **Patch** | 补丁攻击 | 在图像局部区域叠加方形补丁（参考实现，未纳入统一对比） |

## 实验结果（1000 样本）

干净准确率 **99.00%**（990 / 1000 张分类正确）。

**ASR 随扰动预算 ε 的变化：**

| 算法 | ε=0.05 | 0.10 | 0.15 | 0.20 | 0.25 | 0.30 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| FGSM（单步基线） | 4.04% | 13.64% | 30.91% | 55.25% | 74.95% | 88.69% |
| **PGD（迭代）** | 4.14% | 22.22% | **64.34%** | **93.64%** | **99.70%** | **100.00%** |
| **MI-FGSM（迭代）** | 3.94% | 19.09% | 55.86% | 89.39% | 98.99% | 100.00% |

**无固定 ε 的攻击与扰动、耗时对比：**

| 算法 | ASR | 平均 ℓ∞ | 平均 ℓ₂ | 平均耗时 (ms) |
|------|:---:|:---:|:---:|:---:|
| FGSM（ε=0.2） | 55.25% | 0.2000 | 3.3735 | 5.5 |
| PGD（ε=0.2） | 93.64% | 0.2000 | 3.3884 | 57.9 |
| MI-FGSM（ε=0.2） | 89.39% | 0.2000 | 3.4758 | 59.5 |
| C&W | 83.54% | 0.1622 | 2.1038 | 445.2 |
| DeepFool | 51.92% | 0.3877 | 1.5192 | 597.6 |

**核心结论：**
1. 相同扰动预算下，**迭代攻击显著优于单步攻击**（ε=0.2 时 PGD 较 FGSM 提升 38.39 个百分点）；
2. C&W 以更小的 ℓ₂ 扰动（2.10）实现 83.54% 的攻击成功率；
3. DeepFool 的 ℓ₂ 扰动最小（1.52），但受 30 次迭代预算限制，ASR 偏低；
4. 迭代/优化式攻击耗时约为单步攻击的 10~100 倍，收益与代价并存。

## 复现实验

```bash
cd /home/wuchenxi/projects/AIsecurity
/home/wuchenxi/.conda/envs/mixkv/bin/python compare_attacks.py --samples 1000 > outputs/run_1000.log 2>&1
```

## 注意事项

- 本机共享 A100 GPU 上 cuDNN 会初始化失败（`CUDNN_STATUS_NOT_INITIALIZED`），`compare_attacks.py` 已通过 `torch.backends.cudnn.enabled = False` 规避，不影响结果。
- 除 `compare_attacks.py` 外的各 `*.py` 为早期参考实现，部分文件内硬编码了 Windows 路径（如 `D:\pycharm project\...`），在本机运行时需自行修改模型路径；建议统一使用 `compare_attacks.py`。
- 结果图表使用中文标签，已配置 `Noto Sans CJK` 字体，无字体环境时可能出现中文乱码（不影响 CSV 数据）。
