# FIT5217 Assignment 2 — Report Template

**Name:** [Your full name]  
**Student ID:** [Your Monash student ID]   
**Maximum:** 12 pages excluding references

---

## Task 1.4 — Analysis and Reflection

### (1) Hyperparameter Choices

> Describe the hyperparameter values you used and justify each choice.
> Support your justification with experimental evidence (e.g. a small ablation or comparison run).

| Hyperparameter | Value | Justification |
|---|---|---|
| Embedding dimension | | |
| Hidden dimension | | |
| Number of layers | | |
| Learning rate | | |
| Batch size | | |
| Teacher forcing ratio | | |
| Vocabulary frequency threshold | | |
| Early stopping criterion | | |

**Discussion** (≈150 words):

[Write your justification here.]

---

### (2) Baseline vs. Attention Comparison

#### Quantitative Comparison

| Model | BLEU-4 | METEOR | BERTScore |
|---|---|---|---|
| T1.2 Baseline RNN | | | |
| T1.3 Attention RNN | | | |

#### Qualitative Comparison

> Provide 1–2 example ingredient inputs with the generated recipe from each model.
> Discuss where attention helps and where both models fail.

**Example 1**

- **Input (ingredients):** [paste ingredients]
- **Gold recipe:** [paste first 2–3 steps]
- **Baseline output:** [paste model output]
- **Attention output:** [paste model output]

**Discussion** (≈150 words):

[Where does attention help? Are there cases where both models fail similarly?]

---

### (3) Training Dynamics

> Ground your discussion in the loss-curve plots included in your notebook.

**Baseline RNN:**

| | Value |
|---|---|
| Epochs trained | |
| Early stopping triggered at epoch | |
| Final training loss | |
| Final dev loss | |

**Attention RNN:**

| | Value |
|---|---|
| Epochs trained | |
| Early stopping triggered at epoch | |
| Final training loss | |
| Final dev loss | |

**Discussion** (≈150 words):

[Did dev loss correlate with test performance? Did you observe overfitting? Reference your loss-curve figures.]

---

## Task 2.3 — Analysis and Reflection

### (a) Cross-Model Comparison

#### Results Table

| Model | BLEU-4 | METEOR | BERTScore | Rank |
|---|---|---|---|---|
| T1.2 Baseline RNN | | | | |
| T1.3 Attention RNN | | | | |
| T2.1 T5-small + LoRA | | | | |
| T2.1 T5-base + LoRA *(if applicable)* | | | | |
| T2.2 GPT-2 + LoRA | | | | |

#### Discussion (≈200 words)

Address the following points:

- **Pre-trained vs. from-scratch:** Why do pre-trained models (T2) outperform (or underperform) the RNN trained from scratch (T1)?
- **Architectural differences:** How does the encoder–decoder design of T5 differ from the decoder-only design of GPT-2? How do these differences explain any performance gap?
- **Effect of model size** *(if T5-base was run):* How did increasing model size affect performance and compute cost?

[Write your discussion here.]

---

### (b) LoRA Configuration Analysis

#### LoRA Hyperparameters

| Parameter | T5 value | GPT-2 value | Effect on trainable params |
|---|---|---|---|
| `lora_r` | | | |
| `lora_alpha` | | | |
| `lora_dropout` | | | |
| Target modules | | | |

**Trainable parameter fraction:**

| Model | Total params | Trainable (LoRA) | Fraction (%) |
|---|---|---|---|
| T5-small + LoRA | | | |
| GPT-2 + LoRA | | | |

**Discussion** (≈150 words):

[Explain the role of each hyperparameter. Justify your choice of target modules (why attention layers rather than feed-forward layers?). Compare parameter efficiency across the two models.]

---

### (c) Training Dynamics

> Ground your discussion in the loss plots from your notebook.

| Model | Epochs | Early stop at | Final train loss | Final dev loss |
|---|---|---|---|---|
| T5-small + LoRA | | | | |
| GPT-2 + LoRA | | | | |

**Discussion** (≈100 words):

[Did dev loss serve as a reliable proxy for test performance? Were there signs of overfitting? Reference your plots.]

---

## Task 3.4 — Analysis and Reflection

### (a) RAG Pipeline Design

#### Knowledge Base Construction

| Item | Your answer |
|---|---|
| Data split(s) indexed | |
| Preprocessing applied | |
| Number of indexed documents | |
| Average document length (tokens/words) | |
| Indexing time | |

#### Indexing

| Item | Your answer |
|---|---|
| Method (sparse / dense / hybrid) | |
| Library / model used | |
| Key parameters (e.g. BM25 k₁, b; or embedding dim) | |

#### RAG Design

| Item | Your answer |
|---|---|
| Query understanding strategy | |
| Retrieval strategy | |
| Post-retrieval filtering / re-ranking | |
| Prompt construction strategy | |
| Any additional design choices | |

**Architecture diagram** (optional but recommended):

> [Insert flowchart here: query → retrieval → generation → judge. A hand-drawn photo, draw.io export, or any image format is acceptable.]

#### LLM as Judge Design

Present your designed evaluation criteria and rubric for the LLM-as-judge, including the specific criteria you used to evaluate the generated menus, the scoring scale (e.g. 1–5), and any instructions or guidelines you provided to the LLM to ensure consistent and fair evaluation.

**Discussion** (≈150 words):

[Describe design choices and justify how they handle the challenges in this task. For example, how to ensure retrieved recipes are relevant? How to satisfy constraints?]

---

### (b) Per-Level Evaluation

Repeat the block below for each of the three difficulty levels.

---

#### Level 1 — Simple

**Query used:**

> [Paste the Level 1 query you used]

**Top retrieved recipes:**

| # | Title | Brief summary |
|---|---|---|
| 1 | | |
| 2 | | |
| 3 | | |

**Generated menu (full text):**

> [Paste the full LLM-generated menu here]

**Judge scores:**

| Criterion | Self-eval score | Self-eval reasoning | Cross-eval score | Cross-eval reasoning |
|---|---|---|---|---|
| Constraint Satisfaction | | | | |
| Ingredient Faithfulness | | | | |
| Culinary Logic & Coherence | | | | |
| Bias | | | | |

**Reflection** (≈100 words):

[Did the system perform well? Where did it fail? Did the judge's assessment align with your own judgment?]

---

#### Level 2 — Intermediate

**Query used:**

> [Paste the Level 2 query you used]

**Top retrieved recipes:**

| # | Title | Brief summary |
|---|---|---|
| 1 | | |
| 2 | | |
| 3 | | |

**Generated menu (full text):**

> [Paste the full LLM-generated menu here]

**Judge scores:**

| Criterion | Self-eval score | Self-eval reasoning | Cross-eval score | Cross-eval reasoning |
|---|---|---|---|---|
| Constraint Satisfaction | | | | |
| Ingredient Faithfulness | | | | |
| Culinary Logic & Coherence | | | | |
| Bias | | | | |

**Reflection** (≈100 words):

[Reflection here.]

---

#### Level 3 — Hard

**Query used:**

> [Paste the Level 3 query you used]

**Top retrieved recipes:**

| # | Title | Brief summary |
|---|---|---|
| 1 | | |
| 2 | | |
| 3 | | |

**Generated menu (full text):**

> [Paste the full LLM-generated menu here]

**Judge scores:**

| Criterion | Self-eval score | Self-eval reasoning | Cross-eval score | Cross-eval reasoning |
|---|---|---|---|---|
| Constraint Satisfaction | | | | |
| Ingredient Faithfulness | | | | |
| Culinary Logic & Coherence | | | | |
| Bias | | | | |

**Reflection** (≈100 words):

[Reflection here.]

---

#### Retrieval Approach Justification (≈100 words)

[Briefly justify your retrieval choice (sparse / dense / hybrid). When would each approach be preferable?]

---

### (c) Critical Reflection on LLM-as-Judge

> Discuss at least **two failure modes** and at least **one success and one failure** from your experiments.

**Failure mode 1** (≈100 words):

[E.g. position bias, sycophancy, inability to verify factual constraints. Cite relevant papers.]

**Failure mode 2** (≈100 words):

[Another failure mode.]

**Example where the judge succeeded:**

> [Describe the menu and explain why the judge's evaluation was accurate.]

**Example where the judge was wrong or inconsistent:**

> [Describe the menu and explain what the judge got wrong.]

**How to improve judge reliability** (≈100 words):

[E.g. stronger LLMs, multi-sampling, rubric calibration, human-in-the-loop spot-checking.]

---

## References

[Add any references cited in your report here, following a consistent citation style.]
