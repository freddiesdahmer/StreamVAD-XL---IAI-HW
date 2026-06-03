# StreamVAD-XL

使用结合了 CLIP 视觉特征的 Transformer-XL 记忆架构实现实时视频异常检测。本项目基于 [StreamVAD](https://github.com/Han-lijun/StreamVAD) 构建，通过有状态的时间模型（`StreamVADWithXL`）以及一键部署脚本对原项目进行了扩展。

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
| `TransformerXLMemoryBlock` | 具有分离记忆缓存 (detached memory cache) 的单层注意力层 |
| `EnhancedPCI` | 上述组件的 N 层堆叠；负责管理记忆列表 |
| `StreamVADWithXL` | 完整模型：KCG → PCI → LayerNorm → 分类器 |

---

## 项目结构 (Project Structure)

```
StreamVAD_XL/
├── model.py                 # StreamVADWithXL (需手动放置)
├── dataset.py               # UCF / XD 数据集加载器 (需手动放置)
├── realtime_inference.py    # 实时检测入口点 
├── realtime_input.py        # 视频流读取器
├── clip_extractor.py        # CLIP 特征提取器
├── test_xl.py               # 离线 AUC 评估
├── weights/
│   └── xl_best.pth          # 预训练权重 (需手动放置)
├── features/                # 预先提取的 .npy 特征
└── data/                    # CSV 索引文件

```

---

## 快速入门 (Quick Start)

### 环境要求 (Requirements)

* Python 3.8 – 3.10
* Git

### 步骤 1 — 部署 (Deploy)

下载 `deploy_streamvad_xl.py` 并运行：

```bash
python deploy_streamvad_xl.py

```

该脚本会自动执行以下操作：

1. 验证 Python 版本 (3.8.0 – 3.10.12)
2. 检查 Git 是否安装
3. 创建 `StreamVAD_XL/` 项目目录
4. 安装所有依赖项（PyTorch, CLIP, OpenCV 等）
5. 生成 `realtime_inference.py`、`realtime_input.py` 和 `clip_extractor.py`
6. 预下载 CLIP ViT-B/16 权重
7. 打印后续步骤指南

### 步骤 2 — 放置模型文件与权重

将您的模型代码复制到项目目录中：

```
StreamVAD_XL/
├── model.py       ← StreamVADWithXL, EnhancedPCI, TransformerXLMemoryBlock
└── dataset.py     ← UCFDataset_Whole_test_train, XDDataset_Whole_test_train, …

```

放置预训练权重：

```
StreamVAD_XL/weights/xl_best.pth

```

### 步骤 3 — 运行 (Run)

```bash
cd StreamVAD_XL
python realtime_inference.py

```

按 **`q`** 键退出。退出时将打印会话摘要（session summary）。

#### 修改视频源

打开 `realtime_inference.py` 并修改 `VIDEO_SOURCE` 变量：

```python
VIDEO_SOURCE = 0                          # 默认摄像头
VIDEO_SOURCE = r"C:\Videos\test.mp4"     # 本地视频文件

```

---

## 离线评估 (Offline Evaluation)

使用 `test_xl.py` 在预先提取的特征数据集上计算视频级 AUC：

```bash
python test_xl.py \
  --dataset ucf \
  --test_list /path/to/test.csv \
  --model_path weights/xl_best.pth \
  --attnwindow 256 \
  --chunk_size 16

```

| 参数 (Argument) | 默认值 (Default) | 描述 (Description) |
| --- | --- | --- |
| `--dataset` | `ucf` | `ucf`  |
| `--test_list` | — | 测试集 CSV 文件路径 |
| `--model_path` | — | `.pth` 权重文件路径 |
| `--mem_len` | `32` | Transformer-XL 记忆长度 |
| `--chunk_size` | `16` | 每个推理块的帧数  |
| `--attnwindow` | `256` | 截断长度 |
| `--batch_size` | `1` | Dataloader 批大小 |

CSV 文件应包含以下列：`path`（`.npy` 特征文件路径）和 `label`（`Normal` 或异常类别名称）。

---

## 依赖项

| 包名 (Package) | 用途 (Purpose) |
| --- | --- |
| `torch` / `torchvision` | 模型与张量操作 (Tensor ops) |
| `openai/clip` | ViT-B/16 视觉特征提取 |
| `opencv-python` | 视频捕获与 HUD 界面渲染 |
| `Pillow` | 适用于 CLIP 的图像帧格式转换 |
| `numpy` / `pandas` | 数值计算与数据集加载 |
| `scikit-learn` | AUC 指标计算 (`roc_auc_score`) |
| `ftfy` / `regex` | CLIP 文本分词器 (Tokenizer) 依赖项 |

---

## HUD 界面显示

实时窗口会显示一个覆层面板，包含以下内容：

* 运行时间、帧数、FPS、推理延迟
* 带有颜色编码的异常分数条

---

## 引用


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

本项目继承了原 [StreamVAD](https://github.com/Han-lijun/StreamVAD) 仓库的开源协议。详情请前往原仓库查看。
