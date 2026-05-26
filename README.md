# FIT5217 Assignment 2 — Recipe Generation & Menu Design

Monash FIT5217 NLP Assignment 2. Three deliverables:

- **Task 1 — RNN encoder-decoder** trained from scratch in PyTorch, with and without Bahdanau attention.
- **Task 2 — LoRA fine-tuning** of T5 and GPT-2 from Hugging Face, compared against the Task 1 baselines.
- **Task 3 — Menu Designer** combining retrieval-augmented generation (RAG) over the cooking corpus, a free-tier LLM API, and an LLM-as-judge evaluation loop.

> Detailed task spec, hard constraints, language conventions and grading rubric are documented in [CLAUDE.md](CLAUDE.md).

---

## Environment

### Local development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10+ recommended (matches the default Colab runtime).

### Google Colab

The training runs for Task 1 and Task 2 are intended to be executed on Colab Pro. At the top of every Colab notebook:

```python
!pip install -q -r requirements-colab.txt

from google.colab import drive
drive.mount('/content/drive')

import sys
sys.path.append('/content/drive/MyDrive/fit5217_a2')
```

Persist checkpoints, BM25 / FAISS indices and prediction files under `/content/drive/MyDrive/fit5217_a2/` so they survive runtime shutdowns.

---

## Data

Place the three CSV files (provided by the unit) under `data/raw/`:

| File | Rows (incl. header) | Purpose |
|---|---|---|
| `data/raw/train.csv` | 162,900 | Train models, build vocabulary |
| `data/raw/dev.csv` | 1,066 | Hyperparameter tuning, monitoring |
| `data/raw/test.csv` | 1,082 | Final reported scores only |

Each row contains:

- `Title` — recipe name (string)
- `Ingredients` — JSON list of strings, e.g. `["1 c. brown sugar", "1/2 c. milk"]`
- `Recipe` — JSON list of strings, e.g. `["Combine ingredients.", "Boil 5 min."]`

> The test split must not be used for training, preprocessing, or vocabulary construction. The RAG index in Task 3 may be built over train + dev + test.

---

## Environment variables

Free-tier LLM API keys go into a local `.env` file (gitignored):

```bash
cp .env.example .env
# then edit .env and fill in the keys you need
```

```env
GROQ_API_KEY=...
OPENROUTER_API_KEY=...
NVIDIA_NIM_API_KEY=...
```

Load them via `python-dotenv`. Never hardcode keys in notebooks.

---

## Directory layout

```
fit5217_a2/
├── CLAUDE.md                Project brief for the AI assistant
├── README.md                This file
├── requirements.txt         Full local dependencies
├── requirements-colab.txt   Extras to install on top of Colab's default image
├── eval_metrics.py          Unit-provided evaluation script (unchanged)
├── report_template.md       Unit-provided report template (unchanged)
├── data/
│   ├── raw/                 train.csv, dev.csv, test.csv (gitignored)
│   └── sample/              Small debugging subsets (committed)
├── src/
│   ├── utils.py             Seeding, device detection, logging, Colab check
│   ├── preprocessing.py     Tokenization, vocabulary, encoding (Task 1.1)
│   ├── evaluation.py        BLEU-4 / METEOR / BERTScore wrappers
│   ├── models/              RNN (Task 1) and LoRA wrappers (Task 2)
│   └── rag/                 Retrieval, generation, LLM-as-judge (Task 3)
├── notebooks/               A2_T1_<sid>.ipynb, A2_T2_<sid>.ipynb, A2_T3_<sid>.ipynb
├── configs/                 YAML hyperparameter configs
├── checkpoints/             Model weights (gitignored)
├── outputs/                 Predictions, figures, generated xlsx (gitignored)
└── ai_chats/                Generative-AI chat history PDFs for submission
```

---

## Submission checklist

- [ ] `A2_T1_<sid>.ipynb`, `A2_T2_<sid>.ipynb`, `A2_T3_<sid>.ipynb` — all cells executed, outputs preserved
- [ ] `report_<sid>.pdf` — analysis sections T1.4 / T2.3 / T3.4, ≤ 12 pages
- [ ] `generated_<sid>.xlsx` — test predictions for all four models, rows aligned with `test.csv`
- [ ] `AI_chat_history_<sid>.pdf` (if any generative AI was used)

Bundle the above into a single ZIP and upload to Moodle by **2026-05-26 23:55**.
