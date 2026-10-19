<div align="center">

# CTRPredict

**基于 Criteo 1M 数据集的 CTR 预估练手项目：在统一训练闭环中实现并对比 LR / FM / Wide&Deep / DeepFM 四个模型。**

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-AUC-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

[项目简介](#-项目简介) · [项目结构](#-项目结构) · [快速开始](#-快速开始) · [配置说明](#️-配置说明) · [实验结果](#-实验结果)

</div>

---

## 📖 项目简介

> CTRPredict 是一个面向 CTR（点击率）预估的**练手性质**项目：在 Criteo 1M 数据集上，用同一套数据管道与训练流程，完整实现并对比 LR、FM、Wide&Deep、DeepFM 四个经典模型的训练与评估闭环。

项目重点不在于刷高 AUC，而在于把「数据预处理 → 特征工程 → 模型实现 → 训练/早停/调度 → 评估与可视化 → 推理」这条链路走通，并保持代码结构清晰、模型可插拔。四个模型共享同一份 `models.py` 接口与 `train.py` 训练逻辑，通过 YAML 配置文件切换。

---

## ✨ 功能特性

| 功能 | 说明 |
|---|---|
| **四模型统一实现** | `LR` / `FM` / `Wide_Deep` / `DeepFM` 均继承 `nn.Module`，前向签名统一为 `forward(X_cat, X_dense)` |
| **配置驱动** | 每个模型一份 YAML 配置，`--config` 指定；命令行参数可覆盖配置项 |
| **完整预处理管道** | 分层划分 8:1:1、中位数填充、log1p + z-score 标准化、log2 分桶、类别编码与未知类别兜底 |
| **训练策略** | `BCEWithLogitsLoss` + `AdamW` + `ReduceLROnPlateau`（按验证集 AUC 衰减）+ 早停 |
| **训练可视化** | 自动保存 Loss 曲线与 AUC 曲线，并在 AUC 图上标注测试集结果 |
| **统计量持久化** | 预处理统计量与编码映射保存为 `save/artifact.pkl`，推理时直接复用，避免训练/推理不一致 |
| **独立推理模块** | `inference.py` 提供 `process_and_pack` 与 `predict`，可对任意原始格式 DataFrame 打分 |

---

## 📂 项目结构

```text
CTRPredict/
├── config/
│   ├── lr.yaml
│   ├── fm.yaml
│   ├── wide_deep.yaml
│   └── deepfm.yaml
├── data/
│   └── Criteo_1M_with_nans.csv      # 需自行准备，见「快速开始」
├── network/
│   └── models.py                    # LR / FM / Wide_Deep / DeepFM
├── preprocess/
│   └── data_process.py              # process / load_data
├── utils/
│   └── train.py                     # train / evaluate
├── save/                            # 模型权重、曲线图、artifact.pkl（运行时生成）
├── main.py                          # 训练入口
├── inference.py                     # 推理工具函数
├── requirements.txt
└── LICENSE                          # MIT License
```

---

## 📋 环境要求

- **Python**：3.9+
- **核心依赖**：`torch>=2.0`、`pandas`、`numpy`、`scikit-learn`、`matplotlib`、`pyyaml`、`joblib`
- **硬件**：CPU 即可运行；有 GPU 时 PyTorch 会自动使用（当前代码未显式指定 device，默认 CPU）

---

## ⚡ 快速开始

### 1. 安装依赖

```bash
git clone https://github.com/your-username/CTRPredict.git
cd CTRPredict
pip install -r requirements.txt
```

### 2. 准备数据

在项目根目录下创建一个 data 目录，将 Criteo 1M 数据集（含缺失值的版本）放到 `data/Criteo_1M_with_nans.csv`。数据需包含以下列：

| 列类型 | 列名 | 数量 |
|---|---|---|
| 标签 | `target` | 1 |
| 整数特征 | `intCol_0` ~ `intCol_12` | 13 |
| 类别特征 | `catCol_0` ~ `catCol_25` | 26 |

### 3. 创建输出目录

训练过程会向 `save/` 写入模型、曲线图和 `artifact.pkl`，请先确保目录存在：

```bash
mkdir -p save
```

### 4. 开始训练

```bash
# 使用默认配置训练 DeepFM
python main.py --config config/deepfm.yaml

# 命令行参数覆盖 YAML 中的配置
python main.py --config config/fm.yaml --dim 8 --lr 0.0005 --epochs 20
```

训练结束后，`save/` 下会生成：

```text
save/
├── artifact.pkl                    # 预处理统计量与编码映射
├── best_DeepFM.pt                  # 验证集 AUC 最优的模型权重
├── loss_curve_DeepFM.png
└── auc_curve_DeepFM.png
```

模型权重命名规则为 `best_{类名}.pt`，即 `best_LR.pt`、`best_FM.pt`、`best_Wide_Deep.pt`、`best_DeepFM.pt`。

### 5. 推理

`inference.py` 提供的是可复用的工具函数，下面是一个完整的打分示例：

```python
import joblib
import pandas as pd
import torch

from main import build_model
from inference import process_and_pack, predict

# 1. 加载训练时保存的预处理统计量
artifact = joblib.load("save/artifact.pkl")

# 2. 读取待预测数据（需包含原始 intCol_* 与 catCol_* 列，target 可省略）
df = pd.read_csv("data/Criteo_1M_with_nans.csv")

# 3. 重建模型结构并加载权重（dim 必须与训练时一致）
emb_size_of_every_col = [
    len(artifact["encoders"][col]) + 1 for col in artifact["all_cat_cols"]
]
params = {
    "emb_size_of_every_col": emb_size_of_every_col,
    "cat_feature_cols": len(artifact["all_cat_cols"]),
    "dense_feature_cols": len(artifact["dense_cols"]),
    "dim": 8,
}
model = build_model("deepfm", params)
model.load_state_dict(torch.load("save/best_DeepFM.pt", map_location="cpu"))

# 4. 预处理 + 预测
dataloader = process_and_pack(df, artifact, batch_size=256, num_workers=0)
probs = predict(dataloader, model)   # shape: [num_samples]，取值 0~1
print(probs[:10])
```

> 注意：`process_and_pack` 中的缺失值填充、标准化、编码均复用 `artifact`，因此推理结果与训练阶段严格一致。

---

## ⚙️ 配置说明

每个模型对应 `config/` 下的一份 YAML，字段含义如下：

| 字段 | 说明 | 示例 |
|---|---|---|
| `data_path` | 数据集路径 | `data/Criteo_1M_with_nans.csv` |
| `batch_size` | 批大小 | `256` |
| `num_workers` | DataLoader 子进程数 | `4` |
| `dim` | Embedding 维度（LR 不使用） | `8` |
| `lr` | 学习率 | `0.0005` |
| `weight_decay` | AdamW 权重衰减 | `0.005` |
| `epochs` | 最大训练轮数 | `20` |
| `early_stop_patience` | 验证集 AUC 连续不提升的容忍轮数 | `3` |
| `model_name` | 模型标识：`lr` / `fm` / `wide_deep` / `deepfm` | `fm` |

命令行可覆盖的字段：`--dim`、`--lr`、`--epochs`、`--weight_decay`、`--early_stop_patience`。若命令行未传入，则使用 YAML 中的值。

---

## 🔍 数据处理流程

所有预处理统计量**只在训练集上计算**，再应用到验证集/测试集，避免数据泄露。

1. **划分**：按标签分层，8:1:1 切分为训练/验证/测试集（`random_state=42`）。
2. **整数特征**：
   - 用训练集中位数填充缺失值；
   - 按 `x > 2 ? floor(log2(x)) : 1` 分桶，得到 13 个 `*_bin` 类别特征。
   - `log1p(clip(lower=0))` 变换，得到 13 个 `*_log` 稠密特征；
   - 用训练集 mean/std 做 z-score 标准化（std 为 0 时置 1）；
3. **类别特征**：缺失填充为 `'missing'`，用训练集唯一值建立索引映射；验证/测试集中未见类别统一映射为 `unknown_idx = len(mapping)`。
4. **最终输入**：
   - 类别特征共 **39 列**（26 个原始 `catCol_*` + 13 个分桶列）；
   - 稠密特征共 **13 列**（标准化后的 `*_log`）。
5. **持久化**：中位数、mean/std、编码映射、列名等全部写入 `artifact`，由 `main.py` 保存为 `save/artifact.pkl`。

---

## 🧠 模型说明

| 模型 | 核心结构 | 关键实现细节 |
|---|---|---|
| **LR** | 类别特征 Embedding(dim=1) + 稠密特征 → Linear | 无交叉项，Embedding 维度取 1 即可 |
| **FM** | 一阶 Embedding + 二阶交叉项 + 稠密线性项 + 全局偏置 | 一阶/二阶 Embedding **不共享参数**；一阶初始化为 0，二阶初始化为 `N(0, 0.01)`；Embedding 后接 Dropout(0.2) |
| **Wide&Deep** | Wide：Linear；Deep：MLP（BN + ReLU + Dropout） | 两路共享同一份拼接特征；Embedding 后 Dropout(0.3) |
| **DeepFM** | FM 部分 + Deep 部分，二者共享二阶 Embedding | 初始化策略同 FM；**未加 Embedding Dropout**（实测会掉 AUC） |

关于初始化：FM/DeepFM 的 Embedding 没有使用 PyTorch 默认的 `N(0,1)`。39 个类别特征相加后方差可达 39，sigmoid 后大量数值贴近 0/1，梯度会被少数特征主导，容易陷入局部最优。因此一阶项初始化为 0、二阶项初始化为小随机数。

---

## 📊 实验结果

在 Criteo 1M 数据集上的对比（评价指标为 AUC）：

| 模型 | 最佳轮次 | 验证集 AUC | 测试集 AUC |
|---|---|---|---|
| LR | 10 | 0.7699 | 0.7722 |
| FM | 1 | 0.7833 | 0.7869 |
| Wide&Deep | 4 | 0.7833 | 0.7868 |
| **DeepFM** | 1 | **0.7862** | **0.7894** |

**结论与观察：**

- LR 基本收敛，作为基线表现稳定；
- FM / Wide&Deep / DeepFM 的 AUC 均略高于 LR，且 DeepFM 最优；
- FM 与 DeepFM 在第 2 轮即开始过拟合，后续尝试 Dropout、低频特征过滤等正则手段收益有限。主要原因是**数据量小、模型容量大**，属于该数据集规模下的结构性限制；
- 若换用更大规模的数据集，模型间的差距预计会进一步拉大。

---

## ⚠️ 已知局限

- 本项目定位为**学习与练手**，未追求 SOTA 效果，也未做大规模调参；
- 数据集为 Criteo 1M 子集，样本量有限，深度模型容易过拟合；
- 未实现特征哈希、跨特征组合、负采样等进阶特征工程；
- 训练默认使用 CPU，未做显式的 device 管理与混合精度；
- `inference.py` 仅提供函数级接口，未封装为 CLI 或服务。

---

## 🤝 贡献

欢迎提交 Issue 或 PR。建议流程：

1. Fork 本仓库；
2. 创建分支：`git checkout -b feature/your-feature`；
3. 提交改动：`git commit -m "feat: add your feature"`；
4. 推送并发起 Pull Request。

如果新增模型，请保持与其他模型一致的 `forward(X_cat, X_dense)` 签名，并在 `main.py` 的 `build_model` 与 `params` 分支中补充对应逻辑。

---

## 📄 许可证

本项目基于 [MIT License](LICENSE) 发布。

你可以自由地使用、复制、修改、合并、发布、分发、再授权甚至出售本软件的副本，唯一的要求是在分发时保留原始的版权声明和许可证全文。软件按「原样」提供，作者不对使用后果承担任何责任。详见 [LICENSE](LICENSE) 文件。