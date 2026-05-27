# T3.4 Discussion Drafts

## (a-1) RAG Pipeline Design

### Version A (Conservative)

The RAG pipeline used the full train, development, and test splits to build the knowledge base, which is allowed for Task 3 because the system retrieves rather than trains on labels. As shown in A2_T3 Notebook Cell 5, the KB contains 165,045 recipes, was indexed in 5.66 seconds, has an average document length of 54.13 tokens, and retrieves in about 37.39 ms. I chose BM25 because recipe queries are keyword-rich: ingredients, dish names, and course labels often appear directly in the text. Dense retrieval could capture more conceptual similarity, but it would require more compute and extra tuning. Each document concatenates Title, Ingredients, and Recipe, giving the retriever both short labels and detailed evidence. Stopword filtering reduces generic query words, while plural normalization reduces mismatches such as “recipes” versus “recipe”. The RAG stage then uses top_k=12 plus course-specific sub-retrieval, capped at 24 documents.

### Version B (Analytical)

The RAG design favors precision, transparency, and low compute cost over representational sophistication. BM25 is a strong fit because cooking queries are lexically grounded: “soup”, “chicken”, “cake”, “Italian”, and “vegetarian” directly signal relevant recipes. Dense retrieval might retrieve conceptually related dishes without keyword overlap, but that benefit would come with embedding cost, model choice, and possible opacity. The indexed document format, Title + Ingredients + Recipe, is important because titles capture dish identity, ingredients capture constraints, and recipe steps provide evidence for feasibility. Stopword filtering and plural normalization specifically address conversational menu queries, where words like “give”, “suggest”, and “recipes” can otherwise dominate sparse scoring. A2_T3 Notebook Cell 9 shows the multi-query enhancement: the system retrieves a primary top_k=12 context and adds course-specific probes, capped at 24 deduplicated documents. This turns BM25 from a single-query baseline into a practical multi-constraint retriever without adding dense infrastructure.

## (a-2) LLM as Judge Design

### Version A (Conservative)

The judge design uses two evaluation settings to compare model behavior. The self-judge is Groq `llama-3.3-70b-versatile`, the same model used for generation, while the cross-judge is Groq `llama-3.1-8b-instant`. This makes it possible to observe whether the generator-family judge is more favorable than a different model. The rubric separates evaluation into four dimensions: constraint satisfaction, ingredient faithfulness, culinary logic and coherence, and bias. Using four dimensions is more informative than one overall score because a menu can satisfy the course structure while still having weak source attribution. The 1-5 scale provides enough granularity without being too sparse or difficult to interpret. The prompt requires reasoning before the score, which encourages the judge to explain its decision. Finally, the strict JSON schema makes the results in A2_T3 Notebook Cell 18 and Cell 19 easy to parse, compare, and summarize in tables.

### Version B (Analytical)

The judge setup was designed to expose evaluator disagreement rather than hide it behind a single score. Using the generator model as a self-judge tests whether a capable model can evaluate its own outputs, while the 8B cross-judge probes capability-dependent failure modes. The four dimensions decompose errors: constraint satisfaction checks instruction following, faithfulness checks grounding in retrieved recipes, coherence checks meal plausibility, and bias checks cultural or demographic carelessness. This decomposition matters because LLM judges can be unreliable when asked for a broad global judgment. The reasoning-before-score instruction follows the intuition behind chain-of-thought-style deliberation, encouraging the model to state evidence before committing to a rating. The 1-5 scale balances interpretability and resolution, and the JSON schema makes the evaluation reproducible. As Zheng et al. (2023) show, LLM-as-judge can be useful, but its reliability depends on prompt design, judge strength, and comparison against human judgment.

## (b) Level 1 Reflection

### Version A (Conservative)

Level 1 was the simplest query because it asked for explicit course types: soup starter, chicken main, and cake dessert. The generated menu in A2_T3 Notebook Cell 11 used Tortellini Soup, Marinated Chicken Breasts, and Pineapple Delight Cake, so it broadly satisfied the request. The self-judge gave constraint satisfaction 4, while the cross-judge gave 2, showing early disagreement. The self-judge's low faithfulness score, 2, was useful because it flagged possible source-attribution weakness. My own judgment is that the menu is reasonable and meets the constraints. The cross-judge seems too strict because marinated chicken breasts are clearly a chicken main dish.

### Version B (Analytical)

Level 1 shows that the pipeline handles direct lexical constraints well, but judge reliability is unstable. The query maps neatly onto BM25-friendly terms: soup, chicken, and cake. Tortellini Soup, Marinated Chicken Breasts, and Pineapple Delight Cake therefore form a coherent answer in A2_T3 Notebook Cell 11. The main issue is evaluator disagreement. The self-judge gave constraint satisfaction 4, while the cross-judge gave 2 despite acknowledging the three courses. This suggests the smaller judge was over-penalizing or reasoning inconsistently. The self-judge's faithfulness score of 2 is more credible because it questions source support. Overall, L1 succeeds, but scores require qualitative inspection.

## (b) Level 2 Reflection

### Version A (Conservative)

Level 2 asked for a three-course Italian dinner with starter, pasta main, and dessert. The generated menu in A2_T3 Notebook Cell 12 used Antipasto Salad, Italian Skillet Pasta, and Biscuit Tortoni, giving a consistent Italian theme and satisfying the course structure. The self-judge scored it 5 for constraints, 4 for coherence, and 5 for bias, but only 2 for faithfulness. The cross-judge gave unusually low scores: 2 for constraints and 1 for the other dimensions. This is the largest disagreement among the three levels. My interpretation is that the cross-judge may lack stable Italian culinary calibration, while the self-judge may be more accepting of adaptations.

### Version B (Analytical)

Level 2 exposes the strongest evaluator split. The generated menu is thematically coherent: Antipasto Salad, Italian Skillet Pasta, and Biscuit Tortoni fit a recognizable Italian-style dinner, and the pasta main is explicit. However, A2_T3 Notebook Cell 18 shows that the cross-judge collapsed to very low scores, including 1 for coherence and bias. This seems disproportionate because imperfect citations do not make the meal incoherent or biased. The gap suggests the 8B judge may lack stable cultural and culinary calibration when judging whether a dish is “Italian enough”. L2 is successful as a menu, but unreliable as an automatically judged result.

## (b) Level 3 Reflection

### Version A (Conservative)

Level 3 was the hardest query because it combined vegetarian, low-fat, three-course structure, and avoidance of cheese or cream. The generated menu in A2_T3 Notebook Cell 13 used Vegetarian Vegetable Soup, Vegetarian Stroganoff Adaptation, and Fresh Fruit Salad. The constraint notes explicitly say sour cream, Parmesan, and Cool Whip were removed or avoided. This makes the answer reasonable, especially because Fresh Fruit Salad is naturally suitable for a low-fat dessert. The self-judge gave balanced scores, including 4 for constraints. The cross-judge gave only 2 and claimed cheese or cream remained. My judgment is that the system succeeded, but the cross-judge confused source ingredients with the adapted menu.

### Version B (Analytical)

Level 3 is the most informative stress test because it requires constraint composition and recipe adaptation. Retrieval found relevant vegetarian and fruit recipes, but some sources contained disallowed ingredients. The generator addressed this by removing sour cream and Parmesan from Vegetarian Stroganoff and replacing Cool Whip in the dessert. A2_T3 Notebook Cell 13 therefore shows a constraint-adjusted menu, not just a retrieved one. The self-judge's 4/3/4/5 scores reflect this balance: it accepts the constraints while noting imperfect faithfulness. The cross-judge's claim that cheese or cream remained is source-versus-output confusion. It judged retrieved ingredients rather than adaptation notes. L3 is a RAG generation example and a failed cross-judge example.

## (b) Retrieval Justification

### Version A (Conservative)

I used sparse BM25 retrieval because it matched the time and compute constraints of the assignment. It runs on CPU, indexes quickly, and gives interpretable matches through recipe titles and ingredients. This is useful in a cooking domain where many user constraints are keyword-based, such as “chicken”, “cake”, “soup”, or “vegetarian”. Dense retrieval would be better for conceptual or paraphrased queries, but it would require embeddings, additional storage, and more model choices. Hybrid retrieval would likely be ideal, combining lexical precision with semantic recall, but it adds implementation complexity. For this project, BM25 was sufficient and transparent.

### Version B (Analytical)

Sparse retrieval is defensible because recipe search is highly lexical. Ingredients and course labels are not just surface words; they are often the actual constraints. BM25 therefore provides interpretability: matching titles or ingredients can be inspected directly in A2_T3 Notebook Cell 11-13. Dense retrieval would help when a dish is conceptually relevant but lacks exact keywords, such as an Italian dish without “Italian”. A hybrid system would probably be stronger, using BM25 for high-precision ingredient matching and dense embeddings for recall. Under the time budget, BM25 plus multi-query probes gave a practical balance of speed, transparency, and quality.

## (c) Failure Mode 1: Self-preference / Sycophancy

### Version A (Conservative)

The first failure mode is self-preference or sycophancy. Across the three demos, the self-judge gave an average constraint satisfaction score of about 4.3, while the cross-judge gave an average of 2.0. This 2.3-point gap is systematic rather than isolated. Since the self-judge used `llama-3.3-70b-versatile`, the same model as the generator, it may have been more willing to accept the generated menu's own explanations. The cross-judge was harsher, although not always more accurate. A2_T3 Notebook Cell 18 shows this pattern clearly. The safest conclusion is that self-evaluation is useful but should not be treated as independent evidence of quality.

### Version B (Analytical)

The self-cross gap matches concerns about LLM self-preference. Panickssery et al. (2024) report that LLM evaluators can recognize and favor outputs similar to their own generations. In this project, the self-judge gave average constraint satisfaction 4.3, while the cross-judge gave 2.0, a systematic 2.3-point difference shown in A2_T3 Notebook Cell 18. This may reflect direct sycophancy toward the generated answer, but it may also reflect capability alignment: the 70B model may better understand the generator's adaptation strategy, while the 8B judge may misread it. Either way, self-judging lacks provenance independence. A stronger design would separate generator and judge families and use multiple judges to estimate disagreement.

## (c) Failure Mode 2: Factual Hallucination / Source Confusion

### Version A (Conservative)

The second failure mode is factual hallucination and source confusion in the cross-judge. In demo 1, the cross-judge stated that “Marinated Chicken Breasts does not match chicken main dish,” even though chicken breasts are plainly a chicken main. In demo 3, it claimed that the menu did not avoid cheese or cream, although the generated notes explicitly removed sour cream, Parmesan, and Cool Whip. These examples are shown in A2_T3 Notebook Cell 19. The likely issue is that the judge confused retrieved source ingredients with the final adapted menu. This makes the 8B cross-judge useful as a stress test, but unreliable as a final evaluator without human checking.

### Version B (Analytical)

The cross-judge errors illustrate a broader LLM-as-judge limitation: evaluators can sound confident while making factual mistakes. Zheng et al. (2023) discuss LLM judges as promising but imperfect substitutes for human preference evaluation. Here, the 8B judge made two concrete errors in A2_T3 Notebook Cell 19. It denied that Marinated Chicken Breasts matched a chicken main dish, and it penalized demo 3 for cheese and cream even though the final menu explicitly removed those ingredients. The second error is especially important because RAG menus may adapt retrieved recipes. A judge must distinguish evidence sources from final outputs. Smaller models may lack the reasoning capacity to track that provenance reliably.

## (c) Success Example

- In demo 3, the self-judge's constraint reasoning was successful. It recognized that the menu addressed the vegetarian, low-fat, and no-cheese/no-cream requirements, while also noting that fat content was not numerically quantified. This is a calibrated judgment because it rewards constraint satisfaction without overstating nutrition evidence.

## (c) Failure Example

- In demo 1, the cross-judge incorrectly claimed that Marinated Chicken Breasts did not match the requested chicken main dish. This is factually wrong and shows that the judge can generate confident but invalid criticism. Demo 3 also shows source-versus-adaptation confusion around cheese and cream.

## (c) Improvement Strategies

### Version A (Conservative)

Judge reliability could be improved in several practical ways. First, use stronger judge models, such as GPT-4o or Claude Sonnet, especially for final reporting. Second, sample multiple judgments and use majority voting or averaging to reduce stochastic noise. Third, calibrate the rubric with anchor examples showing what scores 1, 3, and 5 should look like. Fourth, use human-in-the-loop spot-checks for high-impact or contradictory cases. Finally, the judge should be independent from the generator model, both for capability comparison and provenance independence. These changes would not remove all subjectivity, but they would make the evaluation more stable.

### Version B (Analytical)

A stronger evaluation design should treat LLM-as-judge as an ensemble process rather than a single oracle. Zheng et al. (2023) motivate LLM judges as scalable evaluators, but the disagreements in A2_T3 Notebook Cell 18 show that judge identity matters. A practical improvement is to use stronger models such as GPT-4o or Claude Sonnet, then combine multiple sampled judgments through majority voting or score averaging. Rubric calibration with anchor examples would reduce scale drift by showing concrete examples of poor, adequate, and excellent menus. Human-in-the-loop review should remain for contradictory or high-stakes evaluations. Finally, generator and judge models should be separated to reduce self-preference and improve provenance independence.
