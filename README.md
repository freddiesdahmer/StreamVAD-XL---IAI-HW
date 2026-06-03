# StreamVAD-XL

基于 Transformer-XL 记忆机制的 **视频异常检测（Video Anomaly Detection, VAD）** 复现工程。

- 主流程：使用 **预提取的特征（`.npy`）** 在 **UCF** 测试集上进行离线评估（AUC）。
- 兼容：提供一个可选的 **一键部署脚本** `deploy_streamvad_xl.py`，用于环境检查与依赖安装（以及生成实时推理相关文件，若你需要进一步扩展）。

> 上游参考：本项目基于 [StreamVAD](https://github.com/Han-lijun/StreamVAD) 思路进行改造与实验。

---

## 架构 (Architecture)

```
视频帧 (Video frames)
    │
    ▼
CLIP ViT-B/16          → 逐帧视觉特征 (512维, L2归一化)
    │
    ▼
关键帧生成器            → 作用于视频片段的软注意力权重 (soft attention weights)
(Key-frame Generator)
    │
    ▼
EnhancedPCI            → 具有跨片段记忆的堆叠 Transformer-XL 块
    │
    ▼
分类器 (Classifier)     → 逐帧异常 logits → Sigmoid 分数 ∈ [0, 1]
```

`model.py` 中的核心组件：

| 类名 (Class) | 作用 (Role) |
| --- | --- |
| `TransformerXLMemoryBlock` | 带记忆缓存（detached memory cache）的单层注意力层 |
| `EnhancedPCI` | 多层堆叠；负责管理记忆列表 |
| `StreamVADWithXL` | 完整模型：KCG → PCI → LayerNorm → 分类器 |

---

## 项目结构 (Project Structure)

> 说明：本仓库本身就是工程目录，可直接运行离线评估；不需要额外 `cd StreamVAD_XL`。

```
.
├── README.md
├── requirements.txt
├── model.py
├── dataset.py
├── test_xl.py
├── deploy_streamvad_xl.py
└── xl_best.pth
```

---

## 环境安装 (Install)

### 方式 1：推荐（requirements.txt）

```bash
pip install -r requirements.txt
```

### 方式 2：一键部署（可选）

运行部署脚本：

```bash
python deploy_streamvad_xl.py
```

该脚本会做一些环境检查并安装依赖（并包含生成实时推理相关文件的逻辑，属于可选扩展能力；本仓库的**主复现流程**仍然是下面的离线评估）。

---

## 离线评估（UCF）(Offline Evaluation - UCF)

使用 `test_xl.py` 在 **预先提取的 `.npy` 特征** 上计算视频级 AUC：

```bash
python test_xl.py \
  --dataset ucf \
  --test_list /path/to/test.csv \
  --model_path xl_best.pth \
  --attnwindow 256 \
  --chunk_size 16
```

### CSV 格式

CSV 需要包含两列：

- `path`：特征文件路径（`.npy`）
- `label`：`Normal` 或异常类别名称（非 `Normal` 会被当作异常）

示例：

```csv
path,label
/path/to/features/video_0001.npy,Normal
/path/to/features/video_0002.npy,Abuse
```

### 参数说明

| 参数 (Argument) | 默认值 (Default) | 描述 (Description) |
| --- | --- | --- |
| `--dataset` | `ucf` | 数据集名称（本仓库复现以 `ucf` 为主） |
| `--test_list` | — | 测试集 CSV 文件路径 |
| `--model_path` | — | 权重文件路径（默认可用：`xl_best.pth`） |
| `--mem_len` | `32` | Transformer-XL 记忆长度 |
| `--chunk_size` | `16` | 每次推理的帧块长度 |
| `--attnwindow` | `256` | 统一截断/填充长度（见 `dataset.py` 的 `pad()`） |
| `--batch_size` | `1` | DataLoader 批大小 |
| `--device` | 自动 | `cuda` 优先，否则 `cpu` |

---

## 权重 (Weights)

- 本仓库默认提供：`xl_best.pth`
- 你也可以通过 `--model_path` 指向你自己的权重文件。

---

## 引用 (Citation)

```bibtex
@misc{streamvad,
  author = {Han Lijun},
  title  = {StreamVAD},
  year   = {2024},
  url    = {https://github.com/Han-lijun/StreamVAD}
}
```

---

## 开源协议 (License)

本项目的协议说明请参考上游 [StreamVAD](https://github.com/Han-lijun/StreamVAD)。如需更清晰的开源分发，建议在本仓库根目录补充 `LICENSE` 文件。
