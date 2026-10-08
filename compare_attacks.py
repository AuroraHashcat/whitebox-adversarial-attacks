# -*- coding: utf-8 -*-
"""
统一的对抗攻击对比实验脚本
=================================================
在 MNIST + LeNet 白盒场景下, 公平对比以下攻击算法:
  - FGSM      (单步基线)
  - PGD       (迭代投影 + 随机初始化)
  - MI-FGSM   (动量迭代)
  - C&W       (优化式迭代, L2 版)
  - DeepFool  (最小扰动迭代)

输出:
  - outputs/compare_summary.csv         汇总表(ASR / 扰动 / 耗时)
  - outputs/accuracy_vs_epsilon_compare.png
  - outputs/asr_vs_epsilon_compare.png
  - outputs/perturbation_compare.png
  - outputs/adv_examples_compare.png    对抗样本可视化

用法:
  python compare_attacks.py --samples 500
  python compare_attacks.py --samples 1000 --eps 0.05,0.1,0.15,0.2,0.25,0.3
"""

import os
import csv
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np

import matplotlib
matplotlib.use("Agg")  # 无 GUI 环境也能保存图片
import matplotlib.pyplot as plt

# 中文字体支持(图表标题/标签使用中文)
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 共享 GPU 环境下 cuDNN 可能初始化失败, 关闭 cuDNN 以保证实验稳定运行
torch.backends.cudnn.enabled = False

# ------------------------- 全局配置 -------------------------
MEAN = (0.1307,)
STD = (0.3081,)

EPSILONS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]  # 攻击扰动预算(像素空间 [0,1])
STEPS = 10                 # PGD / MI-FGSM 的迭代步数
CW_C = 10.0                # C&W 的惩罚系数 c
CW_STEPS = 40              # C&W 优化迭代步数
DEEPFOOL_OVERSHOOT = 0.02  # DeepFool 过冲系数
DEEPFOOL_MAX_ITER = 30     # DeepFool 最大迭代次数

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "lenet_mnist_model.pth")
OUTDIR = os.path.join(SCRIPT_DIR, "outputs")

normalize = transforms.Normalize(MEAN, STD)


# ------------------------- 模型 -------------------------
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def denorm(batch):
    """反归一化: 归一化空间 -> 像素空间 [0,1]"""
    mean = torch.tensor(MEAN).to(batch.device)
    std = torch.tensor(STD).to(batch.device)
    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)


# ------------------------- 攻击算法 -------------------------
# 统一接口: attack(x_denorm, y, eps) -> x_adv_denorm
#   x_denorm: (1,1,28,28) 像素空间 [0,1] 的张量
#   y       : (1,) 真实标签
#   eps     : 扰动预算; C&W / DeepFool 忽略该参数
# 返回值仍在像素空间 [0,1]

def fgsm(model, x, y, eps, device=None):
    x = x.clone().detach().requires_grad_(True)
    out = model(normalize(x))
    loss = F.nll_loss(out, y)
    model.zero_grad()
    loss.backward()
    grad = x.grad.detach()
    adv = x + eps * grad.sign()
    return torch.clamp(adv, 0.0, 1.0).detach()


def pgd(model, x, y, eps, steps=None, device=None):
    if steps is None:
        steps = STEPS
    if eps <= 0:
        return x.clone().detach()
    alpha = eps / steps
    x_orig = x.clone().detach()
    x_adv = x_orig + torch.empty_like(x_orig).uniform_(-eps, eps)
    x_adv = torch.clamp(x_adv, 0.0, 1.0)
    for _ in range(steps):
        x_adv.requires_grad_(True)
        model.zero_grad()
        out = model(normalize(x_adv))
        loss = F.nll_loss(out, y)
        loss.backward()
        grad = x_adv.grad.detach()
        x_adv = x_adv + alpha * grad.sign()
        x_adv = torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps)
        x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()
    return x_adv


def mifgsm(model, x, y, eps, steps=None, decay=1.0, device=None):
    if steps is None:
        steps = STEPS
    if eps <= 0:
        return x.clone().detach()
    alpha = eps / steps
    x_orig = x.clone().detach()
    x_adv = x_orig + torch.empty_like(x_orig).uniform_(-eps, eps)
    x_adv = torch.clamp(x_adv, 0.0, 1.0)
    g = torch.zeros_like(x_adv)
    for _ in range(steps):
        x_adv.requires_grad_(True)
        model.zero_grad()
        out = model(normalize(x_adv))
        loss = F.nll_loss(out, y)
        loss.backward()
        grad = x_adv.grad.detach()
        grad = grad / (grad.abs().mean() + 1e-8)  # L1 归一化
        g = decay * g + grad
        x_adv = x_adv + alpha * g.sign()
        x_adv = torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps)
        x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()
    return x_adv


def cw(model, x, y, eps=None, c=CW_C, steps=None, lr=0.01, device=None):
    if steps is None:
        steps = CW_STEPS
    x_orig = x.clone().detach()
    delta = torch.zeros_like(x_orig, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)
    best_adv = x_orig.clone()
    best_obj = float("inf")
    for _ in range(steps):
        x_adv = torch.clamp(x_orig + delta, 0.0, 1.0)
        out = model(normalize(x_adv))
        true_logit = out.gather(1, y.view(-1, 1)).squeeze(1)
        masked = out.clone()
        masked[0, y.item()] = -1e10
        other_logit, _ = masked.max(dim=1)
        f = torch.clamp(true_logit - other_logit, min=0.0)          # 攻击成功项
        l2 = torch.sum((x_adv - x_orig) ** 2, dim=(1, 2, 3))        # 扰动最小项
        loss = l2 + c * f
        opt.zero_grad()
        loss.mean().backward()
        opt.step()
        with torch.no_grad():
            if loss.item() < best_obj:
                best_obj = loss.item()
                best_adv = x_adv.detach().clone()
    return best_adv


def deepfool(model, x, y, eps=None, overshoot=DEEPFOOL_OVERSHOOT,
             max_iter=None, device=None):
    if max_iter is None:
        max_iter = DEEPFOOL_MAX_ITER
    x_orig = x.clone().detach()
    with torch.no_grad():
        y0 = int(model(normalize(x_orig)).argmax(dim=1).item())
    x_adv = x_orig.clone()
    r_tot = torch.zeros_like(x_orig)
    for _ in range(max_iter):
        x_adv.requires_grad_(True)
        out = model(normalize(x_adv))[0]
        cur = int(out.argmax().item())
        if cur != y0:
            break
        grad_cur = torch.autograd.grad(out[cur], x_adv,
                                       retain_graph=True, create_graph=False)[0]
        min_val, r_i = None, None
        for k in range(10):
            if k == cur:
                continue
            grad_k = torch.autograd.grad(out[k], x_adv,
                                         retain_graph=True, create_graph=False)[0]
            w_k = grad_k - grad_cur
            f_k = (out[k] - out[cur]).detach()
            w_norm = w_k.view(-1).norm(p=2) + 1e-12
            val = torch.abs(f_k) / w_norm
            if min_val is None or val < min_val:
                min_val = val
                r_i = (torch.abs(f_k) / (w_norm ** 2)) * w_k
        if r_i is None:
            break
        r_tot = r_tot + (1.0 + overshoot) * r_i.detach()
        x_adv = torch.clamp(x_orig + r_tot, 0.0, 1.0).detach()
    return x_adv.detach()


# ------------------------- 统一评估 -------------------------
def evaluate(model, loader, attack_fn, device, eps):
    """在原本分类正确的样本上评估一次攻击。

    返回 (attacked, fooled, mean_linf, mean_l2, mean_time_ms, adv_examples)
    """
    attacked, fooled = 0, 0
    linf_list, l2_list, time_list = [], [], []
    adv_examples = []

    for data, target in loader:
        data, target = data.to(device), target.to(device)

        with torch.no_grad():
            out = model(data)
            pred = out.argmax(dim=1)
        if pred.item() != target.item():     # 只攻击原本分类正确的样本
            continue

        attacked += 1
        x_denorm = denorm(data)              # 像素空间 [0,1]

        t0 = time.time()
        adv = attack_fn(model, x_denorm, target, eps)
        dt = (time.time() - t0) * 1000.0     # 毫秒
        time_list.append(dt)

        diff = (adv - x_denorm).view(-1)
        linf_list.append(diff.abs().max().item())
        l2_list.append(diff.norm(p=2).item())

        with torch.no_grad():
            out_adv = model(normalize(adv))
            pred_adv = out_adv.argmax(dim=1)

        if pred_adv.item() != target.item():
            fooled += 1
            if len(adv_examples) < 5:        # 保存前几个成功攻击样本
                adv_examples.append((pred.item(), pred_adv.item(),
                                     x_denorm.squeeze().detach().cpu().numpy(),
                                     adv.squeeze().detach().cpu().numpy()))

    mean_linf = float(np.mean(linf_list)) if linf_list else 0.0
    mean_l2 = float(np.mean(l2_list)) if l2_list else 0.0
    mean_time = float(np.mean(time_list)) if time_list else 0.0
    return attacked, fooled, mean_linf, mean_l2, mean_time, adv_examples


# ------------------------- 绘图 -------------------------
def plot_curves(rows, outdir):
    # rows: 每项为 dict
    by_eps = {}   # name -> (eps_list, asr_list)
    const = {}    # name -> asr (不依赖 eps 的算法)
    for r in rows:
        if r["eps"] is None:
            const[r["name"]] = r["asr"]
        else:
            by_eps.setdefault(r["name"], ([], []))
            by_eps[r["name"]][0].append(r["eps"])
            by_eps[r["name"]][1].append(r["asr"])

    markers = {"FGSM": "o-", "PGD": "s-", "MI-FGSM": "^-",
               "C&W": "--", "DeepFool": "--"}

    # ASR vs epsilon
    plt.figure(figsize=(6, 5))
    for name, (xs, ys) in by_eps.items():
        plt.plot(xs, ys, markers.get(name, "o-"), label=name, linewidth=2)
    for name, asr in const.items():
        plt.axhline(asr, linestyle="--", alpha=0.6,
                    label=f"{name} (无固定 eps)")
    plt.xlabel("Epsilon")
    plt.ylabel("Attack Success Rate (%)")
    plt.title("ASR vs Epsilon (MNIST, LeNet)")
    plt.ylim(0, 105)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "asr_vs_epsilon_compare.png"), dpi=200)
    plt.close()

    # 攻击后准确率 vs epsilon
    plt.figure(figsize=(6, 5))
    for name, (xs, ys) in by_eps.items():
        plt.plot(xs, [100 - y for y in ys], markers.get(name, "o-"),
                 label=name, linewidth=2)
    for name, asr in const.items():
        plt.axhline(100 - asr, linestyle="--", alpha=0.6,
                    label=f"{name} (无固定 eps)")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs Epsilon (MNIST, LeNet)")
    plt.ylim(0, 105)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "accuracy_vs_epsilon_compare.png"), dpi=200)
    plt.close()

    # 平均扰动对比(柱状图, 取 eps=0.2 附近 + 自适应算法)
    names, linfs = [], []
    for name, (xs, ys) in by_eps.items():
        # 找到最接近 0.2 的 eps
        i = min(range(len(xs)), key=lambda k: abs(xs[k] - 0.2))
        names.append(f"{name}\n(eps={xs[i]})")
        linfs.append(next(r["mean_linf"] for r in rows
                          if r["name"] == name and r["eps"] == xs[i]))
    for name in const:
        names.append(name)
        linfs.append(next(r["mean_linf"] for r in rows if r["name"] == name))
    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, linfs, color="steelblue")
    plt.ylabel("Mean L-inf perturbation (pixel space [0,1])")
    plt.title("Average Perturbation Magnitude")
    for b, v in zip(bars, linfs):
        plt.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "perturbation_compare.png"), dpi=200)
    plt.close()


def plot_examples(examples_dict, outdir):
    """每个算法画 5 个成功攻击样本: 上排原始图像, 下排对抗样本。"""
    n = len(examples_dict)
    if n == 0:
        return
    fig, axes = plt.subplots(2 * n, 5, figsize=(2.5 * 5, 2.4 * 2 * n))
    axes = np.atleast_2d(axes)
    for i, (name, exs) in enumerate(examples_dict.items()):
        for j, (orig, adv_lab, img_o, img_a) in enumerate(exs):
            ax_o = axes[2 * i][j]
            ax_a = axes[2 * i + 1][j]
            ax_o.set_xticks([]); ax_o.set_yticks([])
            ax_a.set_xticks([]); ax_a.set_yticks([])
            if j == 0:
                ax_o.set_ylabel(f"{name}\n原始", fontsize=10)
                ax_a.set_ylabel(f"{name}\n对抗", fontsize=10)
            ax_o.set_title(f"label={orig}", fontsize=9)
            ax_a.set_title(f"pred={adv_lab}", fontsize=9)
            ax_o.imshow(img_o, cmap="gray")
            ax_a.imshow(img_a, cmap="gray")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "adv_examples_compare.png"), dpi=200)
    plt.close()


def clean_accuracy(model, loader, device):
    """计算干净(未攻击)样本的分类准确率。"""
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            pred = model(data).argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return 100.0 * correct / total if total else 0.0


# ------------------------- 主流程 -------------------------
def main():
    global STEPS, CW_STEPS, DEEPFOOL_MAX_ITER

    parser = argparse.ArgumentParser(description="对抗攻击统一对比实验")
    parser.add_argument("--samples", type=int, default=500,
                        help="参与攻击的测试样本数(只统计原本分类正确的样本)")
    parser.add_argument("--eps", type=str,
                        default=",".join(str(e) for e in EPSILONS),
                        help="逗号分隔的扰动预算列表")
    parser.add_argument("--outdir", type=str, default=OUTDIR)
    parser.add_argument("--model", type=str, default=MODEL_PATH)
    parser.add_argument("--steps", type=int, default=STEPS,
                        help="PGD / MI-FGSM 的迭代步数")
    parser.add_argument("--cw-steps", type=int, default=CW_STEPS,
                        help="C&W 的优化迭代步数")
    parser.add_argument("--deepfool-iter", type=int, default=DEEPFOOL_MAX_ITER,
                        help="DeepFool 的最大迭代次数")
    args = parser.parse_args()

    STEPS = args.steps
    CW_STEPS = args.cw_steps
    DEEPFOOL_MAX_ITER = args.deepfool_iter

    epsilons = [float(x) for x in args.eps.split(",") if x.strip()]
    os.makedirs(args.outdir, exist_ok=True)

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    # 数据: 固定取测试集前 N 个样本, 保证各算法对比公平
    test_dataset = datasets.MNIST(
        os.path.join(SCRIPT_DIR, "data"), train=False, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]))
    subset = torch.utils.data.Subset(test_dataset, range(args.samples))
    loader = torch.utils.data.DataLoader(subset, batch_size=1, shuffle=False)

    # 模型
    model = Net().to(device)
    try:
        ckpt = torch.load(args.model, map_location=device, weights_only=True)
    except Exception:
        ckpt = torch.load(args.model, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    model.load_state_dict(ckpt)
    model.eval()
    print(f"Model loaded: {args.model}")
    print(f"Clean accuracy on {args.samples} test samples: "
          f"{clean_accuracy(model, loader, device):.2f}%")

    attackers = [
        ("FGSM", fgsm, True),
        ("PGD", pgd, True),
        ("MI-FGSM", mifgsm, True),
        ("C&W", cw, False),
        ("DeepFool", deepfool, False),
    ]

    rows = []
    examples_dict = {}

    for name, fn, use_eps in attackers:
        if use_eps:
            eps_list = epsilons
        else:
            eps_list = [None]
        for eps in eps_list:
            tag = f"{name}" + (f" (eps={eps})" if eps is not None else "")
            print(f"Running {tag} ...", flush=True)
            attacked, fooled, m_linf, m_l2, m_time, adv_exs = \
                evaluate(model, loader, fn, device, eps)
            asr = 100.0 * fooled / attacked if attacked else 0.0
            rows.append({
                "name": name, "eps": eps, "attacked": attacked,
                "fooled": fooled, "asr": asr,
                "mean_linf": m_linf, "mean_l2": m_l2,
                "mean_time_ms": m_time,
            })
            print(f"  attacked={attacked}, fooled={fooled}, "
                  f"ASR={asr:.2f}%, L_inf={m_linf:.4f}, "
                  f"L2={m_l2:.4f}, time={m_time:.1f}ms/sample")
            if use_eps and eps == 0.2:
                examples_dict[name] = adv_exs
            elif not use_eps:
                examples_dict[name] = adv_exs

    # 输出 CSV
    csv_path = os.path.join(args.outdir, "compare_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=[
            "name", "eps", "attacked", "fooled", "asr",
            "mean_linf", "mean_l2", "mean_time_ms"])
        w.writeheader()
        for r in rows:
            r2 = dict(r)
            r2["eps"] = "" if r2["eps"] is None else r2["eps"]
            r2["asr"] = f"{r2['asr']:.2f}"
            r2["mean_linf"] = f"{r2['mean_linf']:.4f}"
            r2["mean_l2"] = f"{r2['mean_l2']:.4f}"
            r2["mean_time_ms"] = f"{r2['mean_time_ms']:.1f}"
            w.writerow(r2)
    print(f"\nSummary saved to {csv_path}")

    # 绘图
    plot_curves(rows, args.outdir)
    plot_examples(examples_dict, args.outdir)
    print("Figures saved to", args.outdir)


if __name__ == "__main__":
    main()
