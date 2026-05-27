# FIT5217 Assignment 2 — Report Template

<!-- This is the DATA-FILLED draft. All tables are populated with experimental data.
     Discussion sections are marked [TODO] and will be written separately. -->

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

## [TODO: write discussion here, target ~150 words]

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

- **Input (ingredients):** ["1 (8 oz.) pkg. cream cheese", "1 small jar Marshmallow Creme", "1 Tbsp. orange juice"]
- **Gold recipe:** whip cream cheese with mixer until smooth . blend in orange juice . add marshmallow creme . whip again until smooth . chill before serving .
- **Baseline output:** mix together and serve with fresh fruit .
- **Attention output:** mix all ingredients together . serve with fresh fruit .

**Discussion** (≈150 words):

## [TODO: write discussion here, target ~150 words]

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

## [TODO: write discussion here, target ~150 words]

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

## [TODO: write discussion here, target ~200 words]

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

## [TODO: write discussion here, target ~150 words]

---

### (c) Training Dynamics

> Ground your discussion in the loss plots from your notebook.

| Model | Epochs | Early stop at | Final train loss | Final dev loss |
|---|---|---|---|---|
| T5-small + LoRA | 3 | Not triggered | 2.5304 | 2.2858 |
| GPT-2 + LoRA | 3 | Not triggered | 1.9199 | 1.8114 |

**Discussion** (≈100 words):

## [TODO: write discussion here, target ~100 words]

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

## [TODO: write discussion here, target ~150 words]

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
| Constraint Satisfaction | 4 | The generated menu satisfies the user's request for a 3-course dinner with a soup starter, a chicken main dish, and a cake for dessert. The menu includes Tortellini Soup as the starter, Marinated Chicken Breasts as the main course, and Pineapple Delight Cake as the dessert. However, the source recipe evidence does not include a recipe for Pineapple Delight Cake, which raises concerns about the menu's faithfulness to the source recipes. Nevertheless, in terms of constraint satisfaction, the menu meets the user's explicit requirements. | 2 | The generated menu consists of 3 courses: a soup starter, a chicken main dish, and a cake for dessert. This satisfies the user's request for a 3-course dinner menu. However, upon closer inspection, the menu does not follow the user's request for a chicken main dish. The Marinated Chicken Breasts dish is a valid chicken main course, but it does not match the user's request for a chicken main dish. The user's request was not explicitly stated as a constraint, but it was implied. Therefore, the menu partially satisfies the constraint. |
| Ingredient Faithfulness | 2 | The menu's faithfulness to the source recipes is questionable. While the Tortellini Soup and Marinated Chicken Breasts dishes are based on actual source recipes, the Pineapple Delight Cake is not found in the provided source recipe evidence. This lack of evidence for the dessert course raises concerns about the menu's ingredient faithfulness. Additionally, the menu does not provide detailed information about the ingredients used in each dish, making it difficult to assess the faithfulness to the source recipes. | 4 | The generated menu's dishes and claims are grounded in the retrieved source recipes. The Tortellini Soup, Marinated Chicken Breasts, and Pineapple Delight Cake dishes are all based on the retrieved source recipes. However, the source recipe titles are not always cited accurately. For example, the Pineapple Delight Cake recipe is not explicitly mentioned in the retrieved source recipes. Additionally, the adaptations made to the recipes are reasonable and transparent. |
| Culinary Logic & Coherence | 3 | The menu's culinary logic and coherence are generally acceptable. The order of the courses makes sense, with a soup starter followed by a main course and then dessert. The pairing of the dishes is also plausible, as the flavors and textures of the Tortellini Soup and Marinated Chicken Breasts could work well together. However, the lack of information about the ingredients and preparation methods used in each dish makes it difficult to fully assess the menu's culinary logic and coherence. | 3 | The generated menu would work as a coherent meal. The course order, dish pairing, preparation logic, and flavor balance are plausible. The soup starter is followed by a chicken main dish, and the meal is finished with a sweet dessert. However, the menu does not take into account the flavor profiles of the dishes. For example, the Marinated Chicken Breasts dish is a savory dish, but it is followed by a sweet dessert. This could be a problem for some diners. |
| Bias | 5 | The menu appears to be neutral and respectful, avoiding stereotypes, exclusionary assumptions, or culturally careless claims. The dishes selected are relatively generic and do not seem to be associated with any particular cultural or ethnic group. The menu does not make any unsupported cultural generalizations or assumptions about the user's preferences or dietary restrictions. | 4 | The generated menu does not avoid stereotypes or exclusionary assumptions. The menu is neutral and respectful, but it does not take into account any cultural or dietary restrictions. The menu assumes that the diner will enjoy a traditional Western-style meal. However, the menu does not make any culturally careless claims or unsupported cultural generalizations. |

**Reflection** (≈100 words):

## [TODO: write discussion here, target ~100 words]

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
| Constraint Satisfaction | 5 | The generated menu follows the explicit user constraints of having a 3-course Italian dinner with a starter, pasta main, and dessert. The menu specifies Italian-inspired dishes for each course, adhering to the cuisine requirement. However, there are no explicit dietary restrictions or avoided ingredients mentioned in the user request, so it's not possible to evaluate the menu on these aspects. Overall, the menu satisfies the given constraints. | 2 | The generated menu consists of 3 courses: Starter (Antipasto Salad), Main (Italian Skillet Pasta), and Dessert (Biscuit Tortoni). However, the user request specifies a 3-course Italian dinner, but the menu does not explicitly state that it is Italian. The pasta main course constraint is partially satisfied, but the dish is not a traditional Italian pasta dish. The dessert course constraint is partially satisfied, but the dish is not a traditional Italian dessert. |
| Ingredient Faithfulness | 2 | The menu claims to be based on source recipes, but upon examination, there are discrepancies. For example, the 'Antipasto Salad' starter does not have a direct match in the provided source recipes. The 'Italian Skillet Pasta' main course is supported by the 'Italian Skillet Dinner' source recipe, but the dessert 'Biscuit Tortoni' does not have a matching source recipe. The adaptations and citations are not entirely transparent or accurate, indicating a lack of faithfulness to the source recipes. | 1 | The generated menu does not cite any source recipe titles accurately. The source recipes provided do not match the dishes in the generated menu. For example, the Antipasto Salad is not found in the source recipes, and the Italian Skillet Pasta is not a direct adaptation of the Italian Skillet Dinner recipe. The Biscuit Tortoni is not found in the source recipes either. |
| Culinary Logic & Coherence | 4 | The menu presents a coherent sequence of courses with a logical flow from a light starter to a more substantial pasta main course and finishing with a dessert. The dishes seem to be appropriately paired for a dinner party, considering traditional Italian meal structures. However, without detailed recipes or ingredient lists for each dish in the menu, it's challenging to fully assess the culinary logic and coherence. Assuming the dishes are prepared traditionally, the menu appears to work as a coherent meal. | 1 | The generated menu does not provide a coherent meal. The Starter (Antipasto Salad) and Main (Italian Skillet Pasta) courses do not complement each other in terms of flavor and texture. The Dessert (Biscuit Tortoni) course is not a traditional Italian dessert and does not follow a logical progression from the previous courses. The menu does not provide a clear preparation logic or flavor balance. |
| Bias | 5 | The menu avoids stereotypes, exclusionary assumptions, or culturally careless claims. It presents a neutral and respectful selection of Italian dishes without making unsupported cultural generalizations. The choice of dishes seems to be based on traditional Italian cuisine without appropriating or misrepresenting cultural elements. Therefore, the menu demonstrates a high level of neutrality and respect. | 1 | The generated menu does not avoid stereotypes or culturally careless claims. The menu assumes that Italian cuisine is associated with pasta and does not provide any cultural context or diversity. The menu also assumes that a traditional Italian dessert is a Biscuit Tortoni, which is not a well-known Italian dessert. |

**Reflection** (≈100 words):

## [TODO: write discussion here, target ~100 words]

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
| Constraint Satisfaction | 4 | The generated menu satisfies the user's request for a 3-course dinner with a soup, main, and dessert. It also adheres to the dietary restrictions of being vegetarian and low-fat, avoiding cheese and cream. However, the menu does not explicitly state the fat content of each dish, which could be a concern. Overall, the menu meets the explicit user constraints. | 2 | The generated menu consists of 3 courses: soup, main, and dessert. However, the user requested a 3-course dinner, but the menu does not explicitly state that it is a dinner. The menu does satisfy the vegetarian and low-fat constraints, but it does not avoid cheese or cream as requested. The 'Vegetarian Stroganoff Adaptation' dish still contains Parmesan cheese, and the 'Fresh Fruit Salad' contains Cool Whip, which is high in fat. |
| Ingredient Faithfulness | 3 | The menu claims to be based on the retrieved source recipes, but there are some discrepancies. For example, the 'Vegetarian Stroganoff Adaptation' claims to remove sour cream and Parmesan cheese, which is a reasonable adaptation. However, the 'Fresh Fruit Salad' is based on the 'Surprise Fruit Salad' recipe, which includes Cool Whip, but the menu does not provide a clear explanation of how the adaptation was made. The source recipe titles are cited accurately, but the adaptations could be more transparent. | 3 | The generated menu claims to use source recipes, but the citations are not always accurate. The 'Vegetarian Vegetable Soup' dish is based on the 'Vegetarian Vegetable Soup' source recipe, but the 'Vegetarian Stroganoff Adaptation' dish is not based on the 'Vegetarian Stroganoff' source recipe as claimed. The adaptation is reasonable, but the transparency is lacking. The 'Fresh Fruit Salad' dish is inspired by the 'Surprise Fruit Salad' source recipe, but it is not a direct adaptation. |
| Culinary Logic & Coherence | 4 | The menu appears to be a coherent meal, with a logical order of courses and dish pairings. The 'Vegetarian Vegetable Soup' is a suitable starter, followed by the 'Vegetarian Stroganoff Adaptation' as a main course, and finishing with the 'Fresh Fruit Salad' as a dessert. The flavor balance and preparation logic seem plausible, but the menu could benefit from more detailed descriptions of each dish and how they work together as a meal. | 3 | The generated menu does not provide a clear course order or dish pairing. The 'Vegetarian Vegetable Soup' is a suitable starter, but the 'Vegetarian Stroganoff Adaptation' is a main course that may not be suitable for a vegetarian dinner. The 'Fresh Fruit Salad' is a light dessert, but it may not be satisfying as a main course. The preparation logic and flavor balance are plausible, but the menu lacks coherence. |
| Bias | 5 | The menu appears to be neutral and respectful, avoiding stereotypes, exclusionary assumptions, or culturally careless claims. The dishes are described in a straightforward manner, without making any unsupported cultural generalizations. The menu does not seem to favor or exclude any particular cultural or culinary tradition, which is a positive aspect. | 4 | The generated menu does not exhibit any obvious biases or stereotypes. The menu is neutral and respectful, but it does not avoid unsupported cultural generalizations. The menu assumes that a vegetarian dinner consists of a soup, a main course, and a dessert, which may not be a universal assumption. |

**Reflection** (≈100 words):

## [TODO: write discussion here, target ~100 words]

---

#### Retrieval Approach Justification (≈100 words)

## [TODO: write discussion here, target ~100 words]

---

### (c) Critical Reflection on LLM-as-Judge

> Discuss at least **two failure modes** and at least **one success and one failure** from your experiments.

**Failure mode 1** (≈100 words):

## [TODO: write discussion here, target ~100 words]

**Failure mode 2** (≈100 words):

## [TODO: write discussion here, target ~100 words]

**Example where the judge succeeded:**

> demo_3 self-judge correctly recognized vegetarian / low-fat / no-cream constraints and noted that fat content was not quantified.

**Example where the judge was wrong or inconsistent:**

> demo_1 cross-judge incorrectly stated that “Marinated Chicken Breasts does not match chicken main dish,” although chicken breasts are plainly a chicken main dish.

**How to improve judge reliability** (≈100 words):

## [TODO: write discussion here, target ~100 words]

---

## References

## [TODO: add references cited in final report]
