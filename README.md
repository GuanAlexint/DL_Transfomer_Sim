# 基于 Transformer 与 SimCSE 的患者主诉多症状智能识别

一个端到端的深度学习系统，用于从非结构化的患者主诉文本中自动识别 9 种核心抑郁相关症状。

---

## 📌 核心创新

- **自监督预训练**：采用 SimCSE 框架，利用 dropout 增强学习更鲁棒的中文医疗句子表示。
- **多模态融合**：结合文本信息（Transformer Encoder）与结构化特征（年龄、性别、病史标志位）。
- **全流程实现**：从零实现数据加载、词表构建、模型训练、评估的全部环节，不依赖 Hugging Face。

---

## 📊 性能

| 指标 | 数值 |
|---|---|
| Micro-F1 | 0.7019 |
| Macro-F1 | 0.5183 |
| Mean AUC | 0.8912 |

---

## 🛠 技术栈

Python · PyTorch · Jieba · Scikit-learn · NumPy · Pandas

---

## 🏗 模型架构

```
患者主诉文本 (visit_sn)
       │
       ▼
  jieba 分词 → Token IDs
       │
       ▼
  Embedding + Positional Encoding
       │
       ▼
  Transformer Encoder (4层, d_model=256, nhead=8)
       │
       ▼
  Mean Pooling              结构化特征 (10维)
       │                 年龄 / 性别 / 就诊次数
       │                 高血压 / 冠心病 / 心衰等
       │                         │
       │                    MLP (→32维)
       │                         │
       └──────── concat ─────────┘
                      │
              分类头 (→9个标签)
                      │
              BCEWithLogitsLoss
```

---

## 🔄 训练流程

系统按以下五个阶段顺序执行：

```
analyze → build_vocab → simcse → train → eval
```

**Stage 1 — 数据分析**
统计样本量、标签分布、文本长度分布、高频疾病名称，结果保存至 `./analysis/`。

**Stage 2 — 构建词表**
使用 jieba 对全量文本分词，过滤低频词（min_freq=2），词表上限 30,000，输出 `vocab_tokens.txt`。

**Stage 3 — SimCSE 自监督预训练**
同一文本经两次带 dropout 的前向传播生成正样本对，batch 内其余样本作负例，使用 InfoNCE loss 学习句子表示。

**Stage 4 — 有监督多标签分类训练**
加载 SimCSE 预训练权重进行 fine-tune，损失函数为 BCEWithLogitsLoss，按 Micro-F1 保存最优 checkpoint。

**Stage 5 — 评估**
在测试集输出 Micro-F1、Macro-F1、per-label AUC 和 mAP。

---

## ⚡ 快速开始

### 安装依赖

```bash
pip install torch numpy pandas scikit-learn matplotlib jieba tqdm
```

### 准备数据

将数据放至以下路径（支持 JSON 数组或 JSONL 格式）：

```
./data_cleaned/train_data_cleaned.json
./data_cleaned/test_data_cleaned.json
```

每条样本格式示例：

```json
{
  "visit_sn": "患者主诉文本……",
  "age": 65,
  "gender": 1,
  "visit_num": 3,
  "is_hypertension": 1,
  "is_ischaemic_heart": 0,
  "is_heart_failure": 0,
  "is_renal": 0,
  "is_pad": 0,
  "is_dementia": 0,
  "is_cvd": 0,
  "labels": {
    "tired": 1, "sleep difficulty": 0, "appetite decreased": 1,
    "move slowly": 0, "irritable": 0, "cognitive": 0,
    "weight decreased": 0, "weight increased": 0, "dispirited": 1
  }
}
```

### 运行全流程

```bash
# 1. 数据分析
python patient_complaint_pipeline.py --stage analyze

# 2. 构建词表
python patient_complaint_pipeline.py --stage build_vocab --out_dir ./ckpts

# 3. SimCSE 预训练
python patient_complaint_pipeline.py --stage simcse --out_dir ./ckpts/simcse --epochs 5

# 4. 有监督训练
python patient_complaint_pipeline.py --stage train \
  --simcse_ckpt ./ckpts/simcse/simcse.pth \
  --out_dir ./ckpts/supervised --epochs 10

# 5. 评估
python patient_complaint_pipeline.py --stage eval \
  --model_ckpt ./ckpts/supervised/best.pth
```

---

## 🔧 主要参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--stage` | 必填 | `analyze` / `build_vocab` / `simcse` / `train` / `eval` |
| `--data_dir` | `./data_cleaned` | 数据目录 |
| `--out_dir` | `./ckpts` | 输出目录 |
| `--max_len` | `256` | 文本最大 token 长度 |
| `--batch_size` | `32` | 批大小 |
| `--epochs` | `3` | 训练轮数 |
| `--d_model` | `256` | Transformer 隐层维度 |
| `--simcse_ckpt` | `""` | SimCSE 权重路径（train 阶段可选） |
| `--model_ckpt` | `""` | 模型权重路径（eval 阶段使用） |

---

## 💡 亮点

- **从零造轮子**：手写 Transformer Encoder 和 SimCSE，不依赖任何预训练模型库。
- **解决真实痛点**：为电子病历结构化提供自动化工具，提升临床效率。
- **前沿技术落地**：将 SimCSE 成功应用于小样本中文医疗 NLP 场景。
- **轻量易部署**：模型参数量小，推断仅依赖 PyTorch，无额外服务依赖。
