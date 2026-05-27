# FIT5217 Assignment 2 — Report Template

<!-- This is the FULL DRAFT with all sections written.
     Discussion sections use Conservative versions and should be
     reviewed and rewritten in author's own voice. -->

**Name:** Jili Chen  
**Student ID:** 35423757   
**Maximum:** 12 pages excluding references

---

## Task 1.4 — Analysis and Reflection

### (1) Hyperparameter Choices

> Describe the hyperparameter values you used and justify each choice.
> Support your justification with experimental evidence (e.g. a small ablation or comparison run).

| Hyperparameter | Value | Justification |
|---|---|---|
| Embedding dimension | 128 | Compact embedding size for a 6,733-token vocabulary and fast single-epoch training. |
| Hidden dimension | 256 | Large enough for recipe context while keeping GRU training feasible on T4/L4. |
| Number of layers | 1 | One layer was used as a minimal baseline and to reduce overfitting risk. |
| Learning rate | 0.001 | Adam learning rate used consistently for baseline and attention runs. |
| Batch size | 128 | High enough for GPU utilization while fitting memory with max target length 80. |
| Teacher forcing ratio | 1.0 during training forward pass | Decoder inputs are gold shifted target tokens in the implemented teacher-forced training loop. |
| Vocabulary frequency threshold | 5 | Chosen min_freq=5 gives vocab size 6,733 with dev occurrence OOV 0.1352%. |
| Early stopping criterion | Dev loss, patience=3 | Checkpoints selected by development loss with patience-based stopping support. |

**Discussion** (≈150 words):

I kept the baseline and attention models on the same core settings so the comparison would be fair: 128-dim embeddings, 256 hidden units, one GRU layer, Adam at lr=0.001, batch size 128, full teacher forcing, and dev-loss checkpointing with patience 3. These choices fit the Colab T4/L4 budget while leaving enough capacity for the recipe vocabulary.
For the vocabulary threshold, I ran a small ablation across {3, 5, 10} (Notebook Cell 6). Going from min_freq=3 to 5 dropped 1,713 tokens but only added 34 dev OOV occurrences — a clear win. Going further from 5 to 10 dropped another 1,558 tokens but cost 57 OOV occurrences, so the marginal benefit fell off. I settled on min_freq=5, which gave a vocabulary of 6,733 — a typical scale for a word-level NMT baseline (Notebook Cell 19).

---

### (2) Baseline vs. Attention Comparison

#### Quantitative Comparison

| Model | BLEU-4 | METEOR | BERTScore |
|---|---|---|---|
| T1.2 Baseline RNN | 0.0741 | 0.2415 | 0.8745 |
| T1.3 Attention RNN | 0.0905 | 0.2538 | 0.8772 |

#### Qualitative Comparison

> Provide 1–2 example ingredient inputs with the generated recipe from each model.
> Discuss where attention helps and where both models fail.

**Example 1**

- **Input (ingredients):** `["1 lb. ground beef", "1/2 c. onions, cut up (sauteed in butter or oil)", "1 tsp. mustard", "1 can chicken gumbo soup", "2 Tbsp. catsup"]`
- **Gold recipe:** combine all ingredients . cook 30 minutes . serve on warm buns .
- **Baseline output:** brown beef and onions . drain . add remaining ingredients . simmer for 20 minutes .
- **Attention output:** brown meat and onions . add mustard and catsup . simmer for 10 minutes . add remaining ingredients . simmer for 30 minutes .

**Discussion** (≈150 words):

Quantitatively, attention improved every metric: BLEU-4 jumped from 0.0741 to 0.0905 (+22%), METEOR from 0.2415 to 0.2538, and BERTScore from 0.8745 to 0.8772 (Notebook Cell 20). The BLEU gain is by far the largest, which suggests attention helps most at surface n-gram alignment rather than just semantic similarity.

The qualitative side backs this up. In the sloppy-joe sample (Example 1 above), the baseline generates a plausible but generic "brown beef and onions, drain, add remaining," while the attention model produces "brown meat and onions, add mustard and catsup," picking up ingredient-specific tokens that the baseline drops. The attention heatmap for this sample (Notebook Cell 16) shows direct mustard→mustard and catsup→catsup alignment, which is exactly the behaviour Bahdanau attention is supposed to produce.

Both models still fall back on safe verbs like "mix" or "serve" for low-information ingredient lists, so attention helps grounding but does not solve generic generation.

---

### (3) Training Dynamics

> Ground your discussion in the loss-curve plots included in your notebook.

**Baseline RNN:**

| Metric | Value |
|---|---|
| Epochs trained | 10 |
| Early stopping triggered at epoch | Not triggered |
| Final training loss | 2.2342 |
| Final dev loss | 2.2945 |

**Attention RNN:**

| Metric | Value |
|---|---|
| Epochs trained | 8 |
| Early stopping triggered at epoch | Not triggered |
| Final training loss | 2.2142 |
| Final dev loss | 2.2514 |

**Discussion** (≈150 words):

Both runs looked steady. The baseline trained for 10 epochs (final train loss 2.2342, dev loss 2.2945), and the attention model trained for 8 epochs (final train loss 2.2142, dev loss 2.2514). Neither hit early stopping because dev loss was still falling at the last recorded epoch in both cases — see Notebook Cell 9 and Cell 14 for the curves.

Train loss sat below dev loss throughout, but the gap never widened and the dev curves never turned upward, so I do not see overfitting in these runs. One detail worth noting: the attention model started with a higher first-epoch dev loss than the baseline. That makes sense, since it has extra parameters and has to learn alignments from scratch. By the final epoch, attention overtook the baseline on dev loss, which lines up with its better BLEU, METEOR, and BERTScore on test.

---

## Task 2.3 — Analysis and Reflection

### (a) Cross-Model Comparison

#### Results Table

| Model | BLEU-4 | METEOR | BERTScore | Rank |
|---|---|---|---|---|
| T1.2 Baseline RNN | 0.0741 | 0.2415 | 0.8745 | 2 |
| T1.3 Attention RNN | 0.0905 | 0.2538 | 0.8772 | 1 |
| T2.1 T5-small + LoRA | 0.0664 | 0.2172 | 0.8677 | 3 |
| T2.1 T5-base + LoRA *(if applicable)* | N/A | N/A | N/A | N/A |
| T2.2 GPT-2 + LoRA | 0.0534 | 0.2574 | 0.8607 | 4 |

#### Discussion (≈200 words)

Address the following points:

- **Pre-trained vs. from-scratch:** Why do pre-trained models (T2) outperform (or underperform) the RNN trained from scratch (T1)?
- **Architectural differences:** How does the encoder–decoder design of T5 differ from the decoder-only design of GPT-2? How do these differences explain any performance gap?
- **Effect of model size** *(if T5-base was run):* How did increasing model size affect performance and compute cost?

The headline result is counterintuitive: the from-scratch T1.3 attention RNN tops the BLEU-4 board at 0.0905, ahead of T1.2 baseline (0.0741), the best T5 LoRA configuration (config2, 0.0664), and GPT-2 LoRA with beam decoding (0.0534) — see Notebook Cell 15. I think three factors explain why the pretrained models underperform here. First, LoRA adapters only update 0.236% to 0.965% of parameters, so the models cannot drift far from their pretraining distribution. Second, all T2 runs stopped at three epochs because of the compute budget, and the dev losses were still falling, so these are under-trained rather than converged models. Third, BLEU-4 rewards surface n-gram overlap with the gold recipe corpus, which the from-scratch RNNs learn directly; pretrained models trained on broader text trade some of that surface fidelity for general fluency.

The story flips on other metrics. GPT-2 takes the top METEOR (0.2574), which suggests its broader vocabulary gives better semantic coverage even when n-gram match is weaker. T5 sits between GPT-2 and the RNNs on BLEU, plausibly because its encoder-decoder design gives stronger ingredient-to-target alignment than GPT-2's single-stream conditioning. This pattern is consistent with the assignment goal of understanding model behaviour rather than chasing the highest score.

---

### (b) LoRA Configuration Analysis

#### LoRA Hyperparameters

| Parameter | T5 value | GPT-2 value | Effect on trainable params |
|---|---|---|---|
| `lora_r` | config1=8; config2=16; config3=8 | 8 | Higher rank increases adapter capacity and trainable parameters. |
| `lora_alpha` | config1=16; config2=32; config3=16 | 16 | Scaling factor, set to 2× rank in these runs. |
| `lora_dropout` | config1=0.05; config2=0.05; config3=0.10 | 0.05 | Regularizes adapter path without training base weights. |
| Target modules | config1=[q,v]; config2=[q,v]; config3=[q,k,v,o] | [c_attn] | More target modules increase adaptation coverage and trainable parameters. |

**Trainable parameter fraction:**

| Model | Total params | Trainable (LoRA) | Fraction (%) |
|---|---|---|---|
| T5-small + LoRA | 61,096,448 | 589,824 | 0.965 |
| GPT-2 + LoRA | 124,734,720 | 294,912 | 0.236 |

**Discussion** (≈150 words):

The clearest finding from the T5 ablation (Notebook Cell 5) is that capacity matters more than coverage. Config1 (r=8, alpha=16, [q,v], 0.485% trainable) got 0.0556 BLEU-4. Doubling the rank in config2 (r=16, alpha=32, same [q,v], 0.965% trainable) lifted BLEU-4 to 0.0664. Config3 spent the same 0.965% trainable budget differently — keeping r=8 but spreading adapters across [q,k,v,o] with higher dropout — and landed at 0.0608, lower than config2.

I read this as evidence that LoRA adaptation is bottlenecked by the rank of each adapter rather than how many attention sub-modules carry one. Higher rank gives each adapter a richer update direction; spreading the same budget thin across more modules dilutes that signal. I chose attention modules over feed-forward because recipe generation is an alignment-heavy task (ingredients → steps), and attention is where that alignment happens (Hu et al., 2022). For GPT-2, the same r=8 yields only 0.236% trainable params because the base model is twice the size of T5-small.

---

### (c) Training Dynamics

> Ground your discussion in the loss plots from your notebook.
> Reported here for the best T5 configuration (config2, r=16). For full per-configuration histories see Notebook Cells 6 and 11.

| Model | Epochs | Early stop at | Final train loss | Final dev loss |
|---|---|---|---|---|
| T5-small + LoRA | 3 | Not triggered | 2.5304 | 2.2858 |
| GPT-2 + LoRA | 3 | Not triggered | 1.9199 | 1.8114 |

**Discussion** (≈100 words):

All three T2 models were still improving at the end of training (Notebook Cell 6 and Cell 11). The table above reports the best T5 configuration (config2, r=16); its dev loss fell 2.45 → 2.35 → 2.29 across the three epochs, and GPT-2's went 1.89 → 1.84 → 1.81. None of them triggered early stopping — I capped training at three epochs to fit the compute budget, not because the models had converged.

One important caveat: T5 and GPT-2 losses are not on the same scale. T5 loss is computed over seq2seq decoder targets, while GPT-2 loss is computed over recipe-continuation tokens in a single sequence. GPT-2's lower dev loss does not imply better BLEU.

---

## Task 3.4 — Analysis and Reflection

### (a) RAG Pipeline Design

#### Knowledge Base Construction

| Item | Your answer |
|---|---|
| Data split(s) indexed | train + dev + test |
| Preprocessing applied | Stopword filtering; plural normalization; document text = Title + Ingredients + Recipe |
| Number of indexed documents | 165,045 |
| Average document length (tokens/words) | 54.13 tokens |
| Indexing time | 5.66 sec |

#### Indexing

| Item | Your answer |
|---|---|
| Method (sparse / dense / hybrid) | Sparse retrieval |
| Library / model used | BM25 implementation in `src/rag/knowledge_base.py` |
| Key parameters (e.g. BM25 k₁, b; or embedding dim) | BM25 default scoring with stopword filtering and plural normalization; retrieval latency 37.39 ms |

#### RAG Design

| Item | Your answer |
|---|---|
| Query understanding strategy | Parse menu constraints through prompt wording plus lightweight auxiliary query probes. |
| Retrieval strategy | Primary query with top_k=12 plus course-specific probes; deduplicate up to 24 source recipes. |
| Post-retrieval filtering / re-ranking | Deduplication by title and leading ingredients; no learned re-ranker. |
| Prompt construction strategy | Include retrieved recipe context, require JSON menu, exact source recipe titles, and explicit constraint notes. |
| Any additional design choices | API keys read from environment variables; generator/self-judge `llama-3.3-70b-versatile`; cross-judge `llama-3.1-8b-instant`. |

**Architecture diagram** (optional but recommended):

> Query → BM25 retrieval + auxiliary course probes → deduplicated recipe context → Groq LLM JSON menu generation → self/cross LLM-as-judge evaluation

#### LLM as Judge Design

Present your designed evaluation criteria and rubric for the LLM-as-judge, including the specific criteria you used to evaluate the generated menus, the scoring scale (e.g. 1–5), and any instructions or guidelines you provided to the LLM to ensure consistent and fair evaluation.

**Discussion** (≈150 words):

I built the knowledge base over the full train + dev + test split — this is allowed for Task 3 because the system only retrieves, never trains on labels. The result is 165,045 indexed recipes, indexed in 5.66 seconds, with an average document length of 54.13 tokens and retrieval latency around 37 ms (A2_T3 Notebook Cell 5). I went with BM25 rather than a dense retriever for two reasons: cooking queries are keyword-rich (ingredients, dish names, and course labels usually appear verbatim), and BM25 is fast on CPU with no model loading. Each document concatenates Title + Ingredients + Recipe so the retriever has both short labels and detailed evidence to score against. Two small preprocessing steps helped: stopword filtering removes generic query words, and plural normalization avoids "recipes" vs "recipe" mismatches. The RAG stage then issues a primary query with top_k=12 and a few course-specific sub-queries, deduplicated up to 24 documents before prompting.

For the judge, I run two configurations side by side. The self-judge is Groq `llama-3.3-70b-versatile` (the same model used for generation); the cross-judge is the smaller `llama-3.1-8b-instant`. The point of pairing them is to expose any generator-family favouritism. The rubric covers four dimensions — constraint satisfaction, ingredient faithfulness, culinary logic & coherence, and bias — because a menu can satisfy course structure while still failing on source attribution, so one global score would hide that. I use a 1–5 scale (enough granularity without being too sparse) and require the judge to write its reasoning before assigning a score, which discourages careless rating. The JSON schema makes results in A2_T3 Cell 18–19 easy to aggregate.

---

### (b) Per-Level Evaluation

Repeat the block below for each of the three difficulty levels.

---

#### Level 1 — Simple

**Query used:**

> Suggest a 3-course dinner menu: a soup starter, a chicken main dish, and a cake for dessert.

**Top retrieved recipes:**

| # | Title | Brief summary |
|---|---|---|
| 1 | Surprise Fruit Salad | Ingredients: 1 can Del Monte tropical fruit, 2 bananas, 1 orange (sweet), 8 oz. jar maraschino cherries, drained and halved. Recipe cue: Combine all ingredients. |
| 2 | Tortellini Soup | Ingredients: 3 cans chicken broth, 1 (12 oz.) pkg. tortellini, 1 bunch spinach, cleaned and stemmed, Parmesan cheese. Recipe cue: Bring broth to a boil. |
| 3 | Marinated Chicken Breasts | Ingredients: 1-inch ginger root, chopped fine, 1/2 c. teriyaki sauce, 1/2 c. soy sauce, 3/4 c. white wine. Recipe cue: Mix ingredients together. |

**Generated menu (full text):**

> **Starter: Tortellini Soup**
> - Source recipes: Tortellini Soup
> - Constraint notes: This dish satisfies the user's request for a soup starter. It is based on the Tortellini Soup recipe, which provides a clear and simple soup option.
> 
> **Main: Marinated Chicken Breasts**
> - Source recipes: Marinated Chicken Breasts
> - Constraint notes: This dish satisfies the user's request for a chicken main dish. The Marinated Chicken Breasts recipe provides a flavorful and well-rounded main course option.
> 
> **Dessert: Pineapple Delight Cake**
> - Source recipes: Pineapple Delight Cake
> - Constraint notes: This dish satisfies the user's request for a cake dessert. The Pineapple Delight Cake recipe provides a sweet and satisfying dessert option.
> 
> **Overall notes:** The full menu fits together by providing a balanced and varied three-course meal. The Tortellini Soup starter offers a light and savory beginning, followed by the Marinated Chicken Breasts main course, which provides a rich and flavorful centerpiece. Finally, the Pineapple Delight Cake dessert offers a sweet and indulgent conclusion to the meal.

**Judge scores:**

| Criterion | Self-eval score | Self-eval reasoning | Cross-eval score | Cross-eval reasoning |
|---|---|---|---|---|
| Constraint Satisfaction | 4 | Menu structure matches the request: Tortellini Soup, Marinated Chicken Breasts, and Pineapple Delight Cake. However, dessert source attribution is weak because Pineapple Delight Cake is not clearly retrieved. | 2 | Judge recognizes three courses but incorrectly questions the chicken main. This preserves the key inconsistency: Marinated Chicken Breasts is plainly a valid chicken main dish. |
| Ingredient Faithfulness | 2 | Two dishes are grounded in retrieved sources, but Pineapple Delight Cake lacks clear source evidence. The menu also gives limited ingredient detail, weakening faithfulness assessment. | 4 | Judge views all dishes as grounded and adaptations reasonable. It still notes citation weakness, especially that Pineapple Delight Cake is not clearly supported by retrieved sources. |
| Culinary Logic & Coherence | 3 | Course order is logical: soup, chicken main, then dessert. Pairing is plausible, but limited ingredient and preparation detail prevents a stronger coherence judgment overall for the meal. | 3 | Meal order is coherent and plausible. The judge notes possible flavor imbalance because a savory chicken main is followed directly by a sweet dessert course. |
| Bias | 5 | Menu is neutral and respectful, with generic dishes and no unsupported cultural claims. It does not stereotype users or assume sensitive dietary identities or needs. | 4 | Judge sees no major stereotypes, but notes the menu assumes a Western-style meal. It remains mostly neutral and avoids clearly careless cultural claims or exclusions. |

**Reflection** (≈100 words):

Level 1 was the easiest of the three queries because it asked for explicit course types — soup starter, chicken main, cake dessert. The generated menu (Notebook Cell 11) returned Tortellini Soup, Marinated Chicken Breasts, and Pineapple Delight Cake, which broadly hit every constraint.

What's interesting is the disagreement between the two judges. Self-judge gave constraint satisfaction 4; cross-judge gave 2. Self-judge also flagged a low faithfulness score (2), which I think is a fair concern about source attribution — Pineapple Delight Cake doesn't appear cleanly in the retrieved set. My own read is that the cross-judge is being too strict: "Marinated Chicken Breasts" is clearly a chicken main dish, and rating it as constraint violation is hard to defend.

---

#### Level 2 — Intermediate

**Query used:**

> Plan a 3-course Italian dinner (starter, pasta main, dessert) for a dinner party.

**Top retrieved recipes:**

| # | Title | Brief summary |
|---|---|---|
| 1 | Cheesy Pizza Dinner | Ingredients: 1 (14 oz.) pkg. Kraft Deluxe macaroni and cheese dinner, 1 c. pepperoni strips, 1/4 c. black olive slices, 1/4 c. green onion slices. Recipe cue: Prepare dinner as directed on package. |
| 2 | Candle Stick Salad(For Holiday Dinner Party) | Ingredients: 8 washed lettuce leaves *, 8 slices canned pineapple, 4 medium bananas, halved, 2 large green or red bell peppers (4 slices from each). Recipe cue: *Note: |
| 3 | Surprise Fruit Salad | Ingredients: 1 can Del Monte tropical fruit, 2 bananas, 1 orange (sweet), 8 oz. jar maraschino cherries, drained and halved. Recipe cue: Combine all ingredients. |

**Generated menu (full text):**

> **Starter: Antipasto Salad**
> - Source recipes: Antipasto Salad, Antipasto Turkey Salad
> - Constraint notes: This dish satisfies the user's starter course constraint by providing a light and refreshing Italian-inspired salad. It includes a variety of ingredients such as olives, peppers, and salami, which are commonly found in Italian cuisine.
> 
> **Main: Italian Skillet Pasta**
> - Source recipes: Italian Skillet Dinner
> - Constraint notes: This dish satisfies the user's pasta main course constraint by providing a hearty and flavorful Italian skillet pasta dish. It includes penne pasta, ground beef, Italian sausage, and mozzarella cheese, which are all typical ingredients in Italian cuisine.
> 
> **Dessert: Biscuit Tortoni**
> - Source recipes: Biscuit Tortoni(An Italian Dessert)
> - Constraint notes: This dish satisfies the user's dessert course constraint by providing a traditional Italian dessert. It includes evaporated milk, powdered sugar, and macaroon crumbs, which are all typical ingredients in Italian desserts.
> 
> **Overall notes:** The full menu fits together by providing a well-rounded and traditional Italian dining experience. Each course is carefully selected to showcase the flavors and ingredients of Italian cuisine, and the dishes are adapted to fit the user's specific requests.

**Judge scores:**

| Criterion | Self-eval score | Self-eval reasoning | Cross-eval score | Cross-eval reasoning |
|---|---|---|---|---|
| Constraint Satisfaction | 5 | Menu matches the requested Italian three-course structure: starter, pasta main, and dessert. No additional dietary or avoided-ingredient constraints were specified by the user in this query. | 2 | Judge says Italian identity is not explicit enough and treats pasta and dessert constraints as only partially satisfied, despite Italian Skillet Pasta and Biscuit Tortoni. |
| Ingredient Faithfulness | 2 | Source grounding is weak: Antipasto Salad and Biscuit Tortoni are not clearly matched to retrieved recipes, while Italian Skillet Dinner only partially supports the pasta dish. | 1 | Judge finds source citations inaccurate, saying retrieved recipes do not match Antipasto Salad, Italian Skillet Pasta, or Biscuit Tortoni closely enough overall as supporting evidence. |
| Culinary Logic & Coherence | 4 | Courses form a coherent Italian-style progression: light starter, pasta main, dessert. Missing detailed recipes prevents full confidence, but the overall meal structure is plausible enough. | 1 | Judge claims the meal lacks coherence, saying starter, pasta main, and dessert do not complement each other and Biscuit Tortoni is not a traditional Italian dessert. |
| Bias | 5 | Menu is respectful and avoids exclusionary assumptions. It presents Italian dishes without broad cultural stereotypes or unsupported claims about diners, cuisine, or dietary identity overall. | 1 | Judge treats the Italian theme as culturally careless, arguing the menu reduces Italian cuisine to pasta and an uncertain dessert tradition without enough cultural context. |

**Reflection** (≈100 words):

Level 2 asked for a three-course Italian dinner with starter, pasta main, and dessert. The generated menu in A2_T3 Notebook Cell 12 used Antipasto Salad, Italian Skillet Pasta, and Biscuit Tortoni, giving a consistent Italian theme and satisfying the course structure. The self-judge scored it 5 for constraints, 4 for coherence, and 5 for bias, but only 2 for faithfulness. The cross-judge gave unusually low scores: 2 for constraints and 1 for the other dimensions. This is the largest disagreement among the three levels. My interpretation is that the cross-judge may lack stable Italian culinary calibration, while the self-judge may be more accepting of adaptations.

---

#### Level 3 — Hard

**Query used:**

> Create a low-fat, vegetarian 3-course dinner (soup, main, dessert). Avoid using cheese or cream.

**Top retrieved recipes:**

| # | Title | Brief summary |
|---|---|---|
| 1 | Vegetarian Stroganoff | Ingredients: 1/2 can vegetarian dinner steaks, diced, 2 c. precooked noodles, 1 onion, diced, 1 c. sour cream. Recipe cue: Glaze onions and mushrooms sprinkled sparingly with paprika. Add diced burger steak and Parmesan cheese. |
| 2 | Vegetarian Vegetable Soup | Ingredients: 2 c. sliced carrots, 1 c. sliced celery, 1 c. sliced onions, 1 small turnip, peeled and diced (optional). Recipe cue: Place raw ingredients in a 5-quart kettle and cover with salt and peppered water. |
| 3 | Surprise Fruit Salad | Ingredients: 1 can Del Monte tropical fruit, 2 bananas, 1 orange (sweet), 8 oz. jar maraschino cherries, drained and halved. Recipe cue: Combine all ingredients. |

**Generated menu (full text):**

> **Starter: Vegetarian Vegetable Soup**
> - Source recipes: Vegetarian Vegetable Soup
> - Constraint notes: This dish satisfies the user's request for a low-fat, vegetarian starter by using a recipe that is inherently vegetarian and low in fat, avoiding cheese and cream. The 'Vegetarian Vegetable Soup' source recipe provides a clear foundation for this course.
> 
> **Main: Vegetarian Stroganoff Adaptation**
> - Source recipes: Vegetarian Stroganoff
> - Constraint notes: To satisfy the user's request for a low-fat, vegetarian main course, we adapt the 'Vegetarian Stroganoff' recipe by removing the sour cream and Parmesan cheese, which are high in fat, and focusing on the vegetarian dinner steaks, mushrooms, and noodles. This adaptation ensures the dish remains vegetarian and reduces the fat content.
> 
> **Dessert: Fresh Fruit Salad**
> - Source recipes: Surprise Fruit Salad
> - Constraint notes: For the dessert course, we draw inspiration from the 'Surprise Fruit Salad' but modify it to avoid the Cool Whip, which could be high in fat. Instead, we focus on fresh fruits like bananas, oranges, and other seasonal fruits, creating a light and refreshing dessert that is both vegetarian and low in fat.
> 
> **Overall notes:** This menu is designed to provide a balanced and satisfying vegetarian dining experience while adhering to the user's constraints of low fat and no cheese or cream. Each dish is carefully adapted from the source recipes to ensure compliance with these requirements, offering a delightful and healthy culinary journey.

**Judge scores:**

| Criterion | Self-eval score | Self-eval reasoning | Cross-eval score | Cross-eval reasoning |
|---|---|---|---|---|
| Constraint Satisfaction | 4 | Menu meets the requested soup, main, and dessert structure and avoids cheese or cream through adaptations. Main limitation: low-fat status is not quantitatively verified nutritionally. | 2 | Judge recognizes the three courses but claims cheese or cream remain. This conflicts with menu notes explicitly removing sour cream, Parmesan, and Cool Whip from dishes. |
| Ingredient Faithfulness | 3 | Sources are cited accurately, and removing sour cream and Parmesan is a reasonable adaptation. Fruit salad adaptation is less transparent because Cool Whip removal is brief. | 3 | Judge sees partial source grounding but questions whether Vegetarian Stroganoff and Surprise Fruit Salad are direct enough adaptations of the cited recipes in final context. |
| Culinary Logic & Coherence | 4 | Meal order is coherent: vegetable soup, vegetarian main, fruit dessert. Dishes are plausible together, though more preparation detail would strengthen the overall judgment substantially here. | 3 | Judge accepts plausible preparation but questions overall course pairing, especially whether the adapted stroganoff and fruit salad form a fully coherent vegetarian dinner menu together. |
| Bias | 5 | Menu is neutral and respectful, with straightforward vegetarian dishes and constraint notes. It avoids stereotypes, exclusionary assumptions, and unsupported cultural generalizations about diners or cuisine. | 4 | Judge finds no obvious stereotypes, but notes the menu assumes a standard soup-main-dessert structure for vegetarian dining, which may not be universal everywhere culturally speaking. |

**Reflection** (≈100 words):

Level 3 was the hardest query because it combined vegetarian, low-fat, three-course structure, and avoidance of cheese or cream. The generated menu in A2_T3 Notebook Cell 13 used Vegetarian Vegetable Soup, Vegetarian Stroganoff Adaptation, and Fresh Fruit Salad. The constraint notes explicitly say sour cream, Parmesan, and Cool Whip were removed or avoided. This makes the answer reasonable, especially because Fresh Fruit Salad is naturally suitable for a low-fat dessert. The self-judge gave balanced scores, including 4 for constraints. The cross-judge gave only 2 and claimed cheese or cream remained. My judgment is that the system succeeded, but the cross-judge confused source ingredients with the adapted menu.

---

#### Retrieval Approach Justification (≈100 words)

I used sparse BM25 retrieval because it matched the time and compute constraints of the assignment. It runs on CPU, indexes quickly, and gives interpretable matches through recipe titles and ingredients. This is useful in a cooking domain where many user constraints are keyword-based, such as “chicken”, “cake”, “soup”, or “vegetarian”. Dense retrieval would be better for conceptual or paraphrased queries, but it would require embeddings, additional storage, and more model choices. Hybrid retrieval would likely be ideal, combining lexical precision with semantic recall, but it adds implementation complexity. For this project, BM25 was sufficient and transparent.

---

### (c) Critical Reflection on LLM-as-Judge

> Discuss at least **two failure modes** and at least **one success and one failure** from your experiments.

**Failure mode 1** (≈100 words):

The first failure mode is self-preference or sycophancy. Across the three demos, the self-judge gave an average constraint satisfaction score of about 4.3, while the cross-judge gave an average of 2.0. This 2.3-point gap is systematic rather than isolated. Since the self-judge used `llama-3.3-70b-versatile`, the same model as the generator, it may have been more willing to accept the generated menu's own explanations. The cross-judge was harsher, although not always more accurate. A2_T3 Notebook Cell 18 shows this pattern clearly. The safest conclusion is that self-evaluation is useful but should not be treated as independent evidence of quality.

**Failure mode 2** (≈100 words):

The second failure mode is factual hallucination and source confusion in the cross-judge. In demo 1, the cross-judge stated that “Marinated Chicken Breasts does not match chicken main dish,” even though chicken breasts are plainly a chicken main. In demo 3, it claimed that the menu did not avoid cheese or cream, although the generated notes explicitly removed sour cream, Parmesan, and Cool Whip. These examples are shown in A2_T3 Notebook Cell 19. The likely issue is that the judge confused retrieved source ingredients with the final adapted menu. This makes the 8B cross-judge useful as a stress test, but unreliable as a final evaluator without human checking.

**Example where the judge succeeded:**

- In demo 3, the self-judge's constraint reasoning was successful. It recognized that the menu addressed the vegetarian, low-fat, and no-cheese/no-cream requirements, while also noting that fat content was not numerically quantified. This is a calibrated judgment because it rewards constraint satisfaction without overstating nutrition evidence.

**Example where the judge was wrong or inconsistent:**

- In demo 1, the cross-judge incorrectly claimed that Marinated Chicken Breasts did not match the requested chicken main dish. This is factually wrong and shows that the judge can generate confident but invalid criticism. Demo 3 also shows source-versus-adaptation confusion around cheese and cream.

**How to improve judge reliability** (≈100 words):

Judge reliability could be improved in several practical ways. First, use stronger judge models, such as GPT-4o or Claude Sonnet, especially for final reporting. Second, sample multiple judgments and use majority voting or averaging to reduce stochastic noise. Third, calibrate the rubric with anchor examples showing what scores 1, 3, and 5 should look like. Fourth, use human-in-the-loop spot-checks for high-impact or contradictory cases. Finally, the judge should be independent from the generator model, both for capability comparison and provenance independence. These changes would not remove all subjectivity, but they would make the evaluation more stable.

---

## References

- Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural Machine Translation by Jointly Learning to Align and Translate.
- Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). LoRA: Low-Rank Adaptation of Large Language Models.
- Panickssery, A., et al. (2024). LLM Evaluators Recognize and Favor Their Own Generations.
- Papineni, K., Roukos, S., Ward, T., & Zhu, W.-J. (2002). BLEU: a Method for Automatic Evaluation of Machine Translation.
- Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E. P., Zhang, H., Gonzalez, J. E., & Stoica, I. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.
