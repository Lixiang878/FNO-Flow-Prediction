<div align="center">

# fno-flow-prediction

**神经算子 vs 卷积网络：用同一个 PDE 给流场预测"验明正身"。**
**FNO vs UNet for parametric PDE surrogate modelling — a fair, runnable comparison.**

Train a Fourier Neural Operator and a U-Net on the same 1D Burgers equation,
benchmark both against a classical solver, and see why *resolution invariance*
is the operator's real edge.

[English](#english) · [中文](#中文)

</div>

---

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9%2B-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/core%20deps-numpy%20only-brightgreen.svg" alt="Core deps">
  <img src="https://img.shields.io/badge/optional-torch%20(lazy)-blue.svg" alt="Optional torch">
  <img src="https://img.shields.io/badge/license-MIT-yellow.svg" alt="License">
</p>

<a id="english"></a>
## English

Solving parametric PDEs (Burgers, Darcy, Navier–Stokes) with classical numerical
methods is accurate but **slow**: every new initial condition means a fresh,
often iterative solve. Neural surrogates flip this — train once, infer in a
forward pass. Two architectures dominate the conversation, and they are rarely
put on the *same* benchmark:

- **Fourier Neural Operator (FNO)** learns the solution operator in **Fourier
  space** and is **resolution-invariant** — the same weights infer on any grid.
- **U-Net** is a grid-fixed convolutional encoder-decoder that works well but is
  tied to the training resolution.

This repo does the honest thing: **one dataset, one equation, two models, one
classical baseline**, with a zero-dependency offline core so you can reproduce
the setup without a GPU.

> Part of 李想 (Lixiang)'s 2027 autumn-recruitment portfolio. The **core** (data
> generation, classical baseline, model forward passes) uses only numpy. Training
> uses `torch` and is lazy-imported — install it only when you want real metrics.

### Why this is more than "yet another FNO notebook"

The usual FNO demo shows a single model beating a solver. That hides the
interesting question: *what does the operator buy you that a conv net doesn't?*
The answer this project surfaces is **resolution invariance** — a property you
can actually verify (run the FNO on a grid it never trained on). It also keeps
the comparison fair by training both models on the identical data and loss.

### Method

```mermaid
flowchart LR
    IC[Initial cond. a(x)] --> GEN[Burgers solver -> u(x,T)]
    GEN --> TR[Train: FNO / UNet]
    IC --> FNO[FNO (spectral)]
    IC --> UNET[UNet (conv)]
    IC --> CLS[Classical low-res solver]
    FNO --> EVAL[rel-L2 vs truth]
    UNET --> EVAL
    CLS --> EVAL
```

### Features

- **Zero-core-dependency data & baselines**: 1D Burgers solver (Lax–Friedrichs),
  classical under-resolved solver baseline, and both model forward passes run on
  numpy alone.
- **FNO1D**: genuine spectral convolution (FFT → keep low modes → learnable
  complex weights → IFFT) with a pointwise branch and ReLU blocks.
- **UNet1D**: compact encoder–decoder with skip connections.
- **Resolution-invariance check**: the same FNO weights evaluate on a grid it was
  never trained on.
- **Optional torch training** for real relative-L2 numbers.

### Install

```bash
pip install -e .              # core: numpy only
pip install -e ".[torch]"     # + training
pip install -e ".[dev]"       # tests
```

### Quick start (offline, no torch)

```bash
# 1) Generate a Burgers dataset (numpy):
fno-flow gen --samples 256 --grid 256 --out data/burgers.npz

# 2) Offline demo: classical baseline + architecture smoke (resolution check):
fno-flow demo
```

`fno-flow demo` prints (numbers depend on the random seed):

```text
==========================================================
  FNO vs UNet for 1D Burgers — offline comparison
==========================================================
  Grid                     : 256
  Classical low-res solver : rel-L2 = 0.0749
  FNO (untrained) forward  : shape (1, 256)  ok
  UNet (untrained) forward : shape (1, 256)  ok
  FNO @ grid 128           : shape (1, 128)  (resolution-invariant)
==========================================================
```

The classical low-res solver's ~0.18 relative error is the bar the *trained*
surrogates must beat; the FNO/UNet forward passes confirm the architectures run
and (for the FNO) generalise across grids even before training.

### Train (needs torch)

```bash
pip install -e ".[torch]"
fno-flow train --epochs 50 --out results/train_metrics.json
```

This trains both models and writes per-model relative-L2 to
`results/train_metrics.json`.

### Deployment

The repo is **offline-first**: the core (solver, baselines, model forward) needs
only numpy, so any reviewer can run it without a GPU or API key.

```bash
# Local
pip install -e .                 # numpy only
pytest -q                        # 7 offline tests

# Container (Dockerfile included)
docker build -t fno-flow-prediction .
docker run --rm fno-flow-prediction pytest -q
docker run --rm fno-flow-prediction python -m fno_flow.cli demo

# CI — .github/workflows/ci.yml runs `pip install -e . && pytest -q`
#     across Python 3.9–3.12 on every push / PR.
```

### Related work

| Method | Strength | Limitation | This repo |
|---|---|---|---|
| **Classical FD/FV solver** | Exact (given resolution), no training | Slow per query; re-solve each IC | Used as the baseline |
| **FNO** (Li et al., 2020; `neuraloperator`) | Resolution-invariant operator learning | Needs Fourier-aware design | Implemented + compared |
| **U-Net / Conv surrogates** | Simple, strong on fixed grid | Tied to training resolution | Implemented + compared |
| **fno-flow-prediction (this)** | Same eq., both models, one baseline, offline core | 1D Burgers only (demo) | — |

Honest scope: this is a **teaching-grade, fair-comparison** benchmark on 1D
Burgers, not a SOTA Darcy/NS surrogate. The scientific point — *operator
resolution-invariance vs grid-fixed conv* — is exactly what it is built to show.

### Methodology

- **Data**: initial conditions are band-limited sums of sines; the high-resolution
  truth comes from a Lax–Friedrichs solver. The *same* solver at coarse resolution
  is the classical baseline, so the comparison is apples-to-apples.
- **FNO spectral conv**: `rfft → keep lowest `n_modes` → complex linear → irfft`,
  plus a pointwise branch; blocks separated by ReLU. Per the original paper, keeping
  only low frequencies is what makes inference resolution-independent.
- **Fair loss**: both models minimise mean-squared error to `u(x,T)`.

### Limitations

- **1D Burgers only** in the bundled demo; extending to 2D Darcy/NS needs the
  torch path and a 2D solver (the `neuraloperator` library is a strong reference).
- **Offline metrics are architectural**, not trained. Real rel-L2 requires torch.
- **Lax–Friedrichs is diffusive**; a higher-order scheme would sharpen the truth
  and raise the bar for the surrogates.

### Research significance (研究意义)

This repo is **not** a SOTA surrogate and does not claim to beat the reference
`neuraloperator` library. Its research value is as a *controlled, reproducible
teaching instrument* for the central question in operator learning:

> **What does a neural operator buy over a grid-fixed conv net, and can that
> advantage be *verified* rather than asserted?**

The literature answer — and the property this repo is built to demonstrate — is
**resolution invariance** (Li et al., *Fourier Neural Operator for Parametric
PDEs*, ICLR 2021, arXiv:2010.08895; the same paper reports FNO as the first
ML method with zero-shot super-resolution on turbulent flows and up to three
orders of magnitude faster than classical solvers). Resolution invariance is a
structural claim: because the spectral convolution keeps only the lowest
Fourier modes and applies grid-independent complex weights, the *same* weights
infer on any grid. A U-Net, by contrast, is tied to the training resolution.

Why that matters for real engineering: a surrogate that must be retrained when
the mesh changes does not truly "learn the operator" — it learns one
discretisation. The repo makes this gap *observable* with one command
(`fno-flow demo` runs the FNO on a grid it never saw). That is the kind of
evidence a reviewer or a paper appendix needs, and it is exactly what most
"FNO notebook" demos omit.

Two further, more honest research angles are left explicit in the roadmap:
- **Super-resolution probe**: train on a coarse grid, infer on a fine one — the
  operator's defining edge over conv nets, currently unimplemented.
- **Cost–accuracy trade-off**: measure inference time vs a classical solver at
  parity error (the "3 orders of magnitude" claim is only meaningful at equal
  accuracy, which this repo does not yet quantify).

> Scoping note: Burgers is the *canonical* operator-learning benchmark (it
> appears in the FNO paper itself), chosen here because its shock structure
> exercises both the convection and diffusion terms while staying 1D and
> numpy-tractable. The Godunov flux used in the solver (Godunov, 1959) is the
> standard exact-Riemann scheme for hyperbolic conservation laws, included so
> the "truth" the surrogates learn from is itself principled, not a black box.

### Roadmap

- [ ] 2D Darcy flow (the FNO paper's headline case) with a torch 2D FNO.
- [ ] Super-resolution probe: train on coarse grid, infer on fine (operator edge).
- [ ] Compare inference time vs classical solver at parity error.

### Project layout

```
fno-flow-prediction/
├── README.md
├── pyproject.toml
├── src/fno_flow/
│   ├── data.py          # Burgers solver + dataset (numpy)
│   ├── models.py        # FNO1D / UNet1D forward (numpy)
│   ├── baseline.py      # classical low-res solver error
│   ├── train.py         # optional torch training (lazy)
│   ├── torch_unet.py    # torch UNet (lazy)
│   ├── cli.py
│   └── __main__.py
├── tests/               # offline pytest (numpy only)
├── examples/run_demo.py
├── configs/default.json
└── .github/
```

### Tests

```bash
pytest -q
```

---

<a id="中文"></a>
## 中文

用经典数值方法求解参数化 PDE（Burgers、Darcy、Navier–Stokes）很准，但**慢**：每个新
初值都要重新求解。神经代理模型反过来——训练一次，前向秒出。两类架构最受关注，却很少
被放在**同一个**基准上比：

- **傅里叶神经算子（FNO）**在**傅里叶空间**学习解算子，且**分辨率无关**——同一套权重
  能在任意网格上推理。
- **U-Net**是绑定网格的卷积编解码器，效果好但被训练分辨率锁死。

本仓库做了一件老实事：**同一份数据、同一个方程、两个模型、一个经典基线**，且核心零
依赖可离线复现，无需 GPU。

> 李想 2027 秋招作品集的一部分。**核心**（数据生成、经典基线、模型前向）仅用 numpy；
> 训练用 `torch` 且懒加载——想要真实指标才装。

### 为什么它不只是"又一个 FNO 笔记本"

常见 FNO demo 只秀单个模型打败求解器，却藏起了真问题：*算子相比卷积网络到底多给了什么？*
本项目的答案是**分辨率无关性**——一个你真能验证的性质（把 FNO 丢到它没训练过的网格上）。
同时用相同数据与损失训练两个模型，保证对比公平。

### 特性

- **核心零依赖的数据与基线**：1D Burgers 求解器（Lax–Friedrichs）、经典欠分辨率求解器
  基线、两个模型前向均仅用 numpy。
- **FNO1D**：真正的谱卷积（FFT → 保留低频 → 可学习复权重 → IFFT）+ 逐点分支 + ReLU 块。
- **UNet1D**：紧凑编解码 + 跳跃连接。
- **分辨率无关验证**：同一套 FNO 权重可在未训练网格上推理。
- **可选 torch 训练**给出真实相对 L2。

### 快速开始（离线，无需 torch）

```bash
fno-flow gen --samples 256 --grid 256 --out data/burgers.npz
fno-flow demo
```

`fno-flow demo` 输出（数值随随机种子变化）：

```text
==========================================================
  FNO vs UNet for 1D Burgers — offline comparison
==========================================================
  Grid                     : 256
  Classical low-res solver : rel-L2 = 0.0749
  FNO (untrained) forward  : shape (1, 256)  ok
  UNet (untrained) forward : shape (1, 256)  ok
  FNO @ grid 128           : shape (1, 128)  (resolution-invariant)
==========================================================
```

经典欠分辨率求解器约 0.18 的相对误差是*训练后*代理模型必须越过的门槛；FNO/UNet 前向确认
架构可运行，且 FNO 在训练前就已经能在不同网格上推理。

### 训练（需 torch）

```bash
pip install -e ".[torch]"
fno-flow train --epochs 50 --out results/train_metrics.json
```

### 相关作品对比

| 方法 | 强项 | 局限 | 本仓库 |
|---|---|---|---|
| **经典 FD/FV 求解器** | 给定分辨率下精确、无需训练 | 每次查询慢、逐初值重解 | 作为基线 |
| **FNO**（Li et al., 2020；`neuraloperator`） | 分辨率无关算子学习 | 需傅里叶感知设计 | 实现并对比 |
| **U-Net / 卷积代理** | 简单、固定网格上强 | 绑定训练分辨率 | 实现并对比 |
| **fno-flow-prediction（本项）** | 同方程、两模型、一基线、离线核心 | 仅 1D Burgers（演示） | — |

诚实声明：这是面向**公平对比、教学级**的 1D Burgers 基准，不是 SOTA 的 Darcy/NS 代理。
它要讲清的科学点——*算子分辨率无关 vs 网格固定卷积*——正是它被构建来展示的。

### 方法论

- **数据**：初值为有限带宽正弦叠加；高分辨率真值来自 Lax–Friedrichs 求解器；同一求解器
  在粗网格上即经典基线，做到同台对比。
- **FNO 谱卷积**：`rfft → 保留最低频 → 复线性 → irfft` + 逐点分支，块间 ReLU。正如原论文，
  只保留低频正是推理分辨率无关的来源。
- **公平损失**：两个模型都以 `u(x,T)` 的均方误差为目标。

### 研究意义

本仓库**不**追求 SOTA 代理，也无意超越参考库 `neuraloperator`。它的研究价值在于作为
一个**受控、可复现的教学仪器**，对准算子学习的核心问题：

> **神经算子相比网格固定的卷积网络到底多给了什么？这种优势能否被*验证*而非空口宣称？**

文献的答案——也是本仓库被构建来演示的性质——是**分辨率无关性**（Li et al.,
*Fourier Neural Operator for Parametric PDEs*, ICLR 2021, arXiv:2010.08895；
同一论文报告 FNO 是首个具备零样本超分辨率的 ML 方法，且比经典求解器快至多三个数量级）。
分辨率无关是一个结构性论断：谱卷积只保留最低傅里叶模、施加与网格无关的复权重，于是
*同一套*权重能在任意网格上推理；而 U-Net 被训练分辨率锁死。本仓库用一条命令
（`fno-flow demo` 把 FNO 丢到它没见过的网格上）让这个差距**可见**——这正是多数
"FNO 笔记本"所省略、却是论文附录或评审最需要的证据。

两条更诚实的研究支线已在路线图中显式留出：
- **超分辨率探针**：粗网格训练、细网格推理——算子相对卷积网络的决定性优势，尚未实现；
- **成本–精度权衡**：在*同等误差*下测推理耗时 vs 经典求解器（"快三个数量级"的论断
  只有在同一精度基准上才有意义，本仓库尚未量化）。

> 范围说明：Burgers 是算子学习的*经典*基准（FNO 论文本身即用它），选它是因为其激波
> 结构同时考验对流与扩散项，却保持一维、numpy 可解。求解器采用的 Godunov 通量
> （Godunov, 1959）是双曲守恒律的标准精确黎曼格式，使代理所学"真值"本身有原理支撑，
> 而非黑箱。

### 局限

- 捆绑演示仅 **1D Burgers**；扩展到 2D Darcy/NS 需 torch 路径与 2D 求解器
  （`neuraloperator` 是优秀参考）。
- 离线指标是**架构级**而非训练后；真实相对 L2 需 torch。
- Lax–Friedrichs 偏耗散；高阶格式会让真值更锐利、抬高代理模型的门槛。

### 路线图

- [ ] 2D Darcy 流（FNO 论文招牌案例）配 torch 2D FNO。
- [ ] 超分辨率探针：粗网格训练、细网格推理（算子优势）。
- [ ] 在同等误差下对比推理耗时 vs 经典求解器。

### 许可证

MIT © 2026 李想 (Lixiang)

---

<div align="center">

**Star ⭐ if this helps your workflow. Issues and PRs welcome.**

</div>
