# CLAUDE.md — FIT5217 Assignment 2 项目说明

> 给 Claude Code(以及未来的我)读的项目纲领。
> 任何代码生成、修改、分析任务,都应当先回到这里对齐约束。

---

## 1. 作业概述

**课程**:Monash FIT5217 Natural Language Processing
**主题**:Recipe Generation(给定配料列表 → 生成分步骤菜谱)+ Menu Design
**总分**:100(其中 Week 12 面试占 24)
**截止**:2026-05-26 23:55(via Moodle,ZIP 单文件提交)

三大任务:

| 任务 | 分值 | 内容 |
|---|---|---|
| **T1** | 24 | 用 PyTorch 从零实现 **RNN encoder-decoder**,先 baseline 无 attention,再加 Bahdanau attention |
| **T2** | 27 | 用 **LoRA fine-tuning** 微调 T5(≥ 3 种配置)和 GPT-2,与 T1 对比 |
| **T3** | 25 | 构建 Menu Designer:**RAG 检索** + 免费 LLM API 生成 + **LLM-as-judge** 评估(self + cross) |
| **Interview** | 24 | Week 12 现场答辩 + Menu Designer live demo |

---

## 2. 关键约束(必须遵守)

### 数据纪律
- **test split 不得用于任何训练、预处理或词表构建**。Task 1/2 的词表只能从 train 构建。
- Task 3 的 RAG 索引**可以**建在 train + dev + test 全量上(因为 RAG 不在数据上训练)。

### 实现纪律
- **Bahdanau attention 必须自己实现**。
- **禁止使用** `torch.nn.MultiheadAttention`、HuggingFace 内置 attention 模块、任何现成的 `AdditiveAttention`。允许 `torch.nn` / `torch.nn.functional` 中的基础算子。
- 所有随机种子统一设为 **42**(torch / numpy / random / cudnn deterministic 一起设)。
- 所有 LLM API key **必须**走环境变量(`.env` + `python-dotenv`),不得硬编码到 notebook 或源码。

### Colab 预算(共 100 units/月)
| 任务 | 建议消耗 | 推荐 GPU |
|---|---|---|
| T1 | 30–50 units | T4 / L4(A100 不必要) |
| T2 | < 60 units | A100 长跑;调试用 T4/L4 |
| T3 | 剩余 | CPU 即可 |

省 units 的实践:
- 先用 `data/sample/` 小样本跑通管线再上全量
- checkpoint 和索引一律存 Google Drive(`/content/drive/...`),不要放 `/content/`
- 嵌入向量、BM25 索引等持久化到磁盘,避免重算

---

## 3. 数据格式

数据位置:`data/raw/{train,dev,test}.csv`

| 列 | 类型 | 说明 |
|---|---|---|
| `Title` | string | 菜名(T1/T2 不用,T3 RAG 会用) |
| `Ingredients` | JSON list of string | 例:`["1 c. brown sugar", "1/2 c. milk"]` |
| `Recipe` | JSON list of string | 例:`["Combine ingredients.", "Boil 5 min."]` |

| Split | 行数(含表头) | 用途 |
|---|---|---|
| train.csv | 162,900 | 训练 |
| dev.csv | 1,066 | 超参调优 / 监控训练 |
| test.csv | 1,082 | **仅最终评估** |

读 CSV 时要用 `json.loads` 解析 `Ingredients` 和 `Recipe` 两列(存的是 JSON 字符串)。

---

## 4. 评估指标(Task 1 & 2)

使用助教提供的 `eval_metrics.py`(根目录,不要改动)。三项指标:**BLEU-4 / METEOR / BERTScore (F1)**。

- BLEU-4 weights = (0.25, 0.25, 0.25, 0.25),smoothing = `SmoothingFunction().method4`
- METEOR 用 NLTK,需先 `nltk.word_tokenize`
- BERTScore 直接传字符串列表,`lang='en'`
- **优先用 `evaluate2`(corpus-level BLEU)报最终 test 分**
- 脚本顶部 `nltk_cache_dir = [YOUR_DIR]` 是占位符,使用前必须改为真实路径(本地 `./nltk_data`,Colab `/content/nltk_data`)

Task 3 的评估用 LLM-as-judge,四维度(Constraint Satisfaction / Ingredient Faithfulness / Culinary Logic & Coherence / Bias),每条菜单 self-eval + cross-eval 两轮。

---

## 5. 提交要求

四个文件,放进一个 ZIP 提交到 Moodle:

| 文件 | 说明 |
|---|---|
| `A2_T1_<sid>.ipynb` / `A2_T2_<sid>.ipynb` / `A2_T3_<sid>.ipynb`(三个 notebook),**或**单一 `code_<sid>.ipynb` | 所有 cell 必须执行过且输出保留;notebook 顶部要有 GenAI 声明 cell |
| `report_<sid>.pdf` | 写分析部分(T1.4 / T2.3 / T3.4),≤ 12 页(不含 references) |
| `generated_<sid>.xlsx` | test 集预测,列:`Title / Ingredients / Recipe_Gold / Recipe_T1_Baseline / Recipe_T1_Attention / Recipe_T2_T5 / Recipe_T2_GPT2`,行序与 test.csv 一致 |
| `AI_chat_history_<sid>.pdf`(若用了 GenAI) | 完整对话记录,放 `ai_chats/` |

`<sid>` 替换为我的 Monash 学号。

---

## 6. 语言规范(非常重要,Claude 必读)

**核心原则**:对老师可见的内容用**英文**,对我可见的开发辅助内容用**中文**。

| 内容 | 语言 | 理由 |
|---|---|---|
| 与我对话 | **中文** | 我母语中文 |
| `.py` 代码里的 `#` 注释 | **中文** | 方便我读懂代码逻辑 |
| `.py` 函数 docstring(三引号块) | **英文** | docstring 会出现在 notebook 帮助文档里,老师会看到 |
| Notebook 顶部声明 cell(name / sid / GenAI declaration) | **英文** | 老师阅读 |
| Notebook 的 markdown cell(任务说明、结果分析、讨论) | **英文** | 提交内容 |
| `report_<sid>.pdf`(基于 `report_template.md` 填写) | **英文** | 提交内容 |
| `README.md` | **英文** | 项目首页,符合开源惯例 |
| `CLAUDE.md`(本文件) | **中文** | Claude 内部说明,不提交 |
| Git commit message | 英文(若入仓) | 惯例 |

**举个例子**,下面是一段符合规范的代码:

```python
def build_vocab(corpus: list[list[str]], min_freq: int = 5) -> dict[str, int]:
    """Build a word2idx mapping from the training corpus.

    Parameters
    ----------
    corpus : list[list[str]]
        Tokenized sentences from the training split only.
    min_freq : int
        Tokens appearing fewer than ``min_freq`` times are mapped to <UNK>.
    """
    # 统计词频,这里只能用 train,不能碰 dev/test
    counter = Counter(tok for sent in corpus for tok in sent)
    # 四个特殊 token 一定要放在最前面,保证 <PAD>=0
    vocab = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
    ...
```

---

## 7. 项目结构

```
fit5217_a2/
├── CLAUDE.md            ← 本文件(中文)
├── README.md            ← 项目首页(英文)
├── .gitignore
├── requirements.txt          本地开发完整依赖
├── requirements-colab.txt    Colab 上只装额外需要的包
├── eval_metrics.py           助教提供,不改动
├── report_template.md        助教提供,不改动
├── FIT5217_2026_S1_*.pdf     作业说明
├── data/
│   ├── raw/                  原始 train/dev/test.csv(.gitignore 忽略)
│   └── sample/               调试用小样本(可入仓)
├── src/
│   ├── utils.py              set_seed / get_device / setup_logging / is_colab
│   ├── preprocessing.py      T1.1 tokenize / vocab / encode
│   ├── evaluation.py         BLEU/METEOR/BERTScore 封装
│   ├── models/               T1 RNN / T2 LoRA
│   └── rag/                  T3 RAG pipeline / LLM-as-judge
├── notebooks/                A2_T1/T2/T3_<sid>.ipynb
├── configs/                  超参 YAML
├── checkpoints/              .gitignore
├── outputs/                  生成的预测、xlsx、图表(.gitignore)
└── ai_chats/                 AI 对话历史 PDF
```

---

## 8. 我的开发偏好

- **模块化**:核心逻辑放 `src/` 下,notebook 只做"调用 + 展示 + 分析"。
- **Colab 友好**:`src/` 下模块必须能在 Colab 里 `from src.xxx import yyy`(挂 Drive + `sys.path.append`)。
- **不要过度抽象**:三行相似代码好过一个早期抽象。不要为了"看起来专业"加无意义封装。
- **不要过度防御**:内部代码相互信任,只在系统边界(用户输入、外部 API)做校验。
- **报告先列骨架,再填内容**:先把表格和分点列出来对齐结构,再写具体分析。
- **不要主动跑训练**:训练全部在 Colab 上跑,本地只跑预处理和评估这类轻量任务。给可粘贴的 cell 让我自己执行。

---

## 9. 常见任务的 Claude 工作流

| 我说 | Claude 应该 |
|---|---|
| "实现 T1.1 预处理" | 在 `src/preprocessing.py` 写函数(中文 # 注释 + 英文 docstring),notebook 里加调用 cell + 英文 markdown + sanity check |
| "训练 baseline RNN" | 不要自己跑;写好代码,给我可直接粘进 Colab 的 cell |
| "评估模型" | 调用 `src/evaluation.py`,不要重新实现 BLEU/METEOR |
| "更新报告" | 编辑 `outputs/` 下 `report_template.md` 的副本,英文按模板填表 |
