"""
FIT5217 Assignment 2 — Evaluation Script
Computes BLEU-4 (corpus-level), METEOR (sentence-level avg), and BERTScore
for Tasks 1 and 2.

Usage:
    python eval_metrics.py          # runs a quick self-test with dummy data
"""

import nltk, numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction, corpus_bleu
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bert_score

nltk_cache_dir = [YOUR_DIR]
nltk.download('punkt', download_dir=nltk_cache_dir)
nltk.download('wordnet', download_dir=nltk_cache_dir)
nltk.download('omw-1.4', download_dir=nltk_cache_dir)
nltk.download('punkt_tab', download_dir=nltk_cache_dir)
nltk.data.path.append(nltk_cache_dir)

def evaluate1(gold_recipes, pred_recipes):
    bleu4, meteor = [], []
    for gold, pred in zip(gold_recipes, pred_recipes):
        ref = nltk.word_tokenize(gold)
        hyp = nltk.word_tokenize(pred)
        sm = SmoothingFunction().method4
        bleu4.append(sentence_bleu([ref], hyp, weights=(.25,.25,.25,.25),
                     smoothing_function=sm))
        meteor.append(meteor_score([ref], hyp))
    _, _, F1 = bert_score(pred_recipes, gold_recipes, lang='en', verbose=False)
    print(f"BLEU-4:    {np.mean(bleu4):.4f}")
    print(f"METEOR:    {np.mean(meteor):.4f}")
    print(f"BERTScore: {F1.numpy().mean():.4f}")
    
    
def evaluate2(gold_recipes, pred_recipes):
    refs = [[nltk.word_tokenize(gold)] for gold in gold_recipes]
    hyps = [nltk.word_tokenize(pred) for pred in pred_recipes]
        
    sm = SmoothingFunction().method4
    corpus_avg_bleu4_score = corpus_bleu(refs, hyps, weights=(.25,.25,.25,.25), smoothing_function=sm)
    corpus_avg_meteor_score = np.mean([meteor_score(ref, hyp) for ref, hyp in zip(refs, hyps)])
    _, _, F1 = bert_score(pred_recipes, gold_recipes, lang='en', verbose=False)
    print(f"BLEU-4:    {corpus_avg_bleu4_score:.4f}")
    print(f"METEOR:    {corpus_avg_meteor_score:.4f}")
    print(f"BERTScore: {F1.numpy().mean():.4f}")
