"""
Natural Language Generation Evaluation Metrics.
Calculates BLEU-1/2/4, ROUGE-1/2/L, and semantic text similarity.
"""

from typing import Dict, List
import re
from collections import Counter
import math


def compute_ngram_counts(tokens: List[str], n: int) -> Counter:
    """Extracts n-gram counts from token list."""
    if len(tokens) < n:
        return Counter()
    return Counter([tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)])


def compute_bleu(reference: str, candidate: str, max_n: int = 4) -> Dict[str, float]:
    """Calculates BLEU scores with brevity penalty."""
    ref_tokens = re.findall(r"\w+", reference.lower())
    cand_tokens = re.findall(r"\w+", candidate.lower())

    if not cand_tokens or not ref_tokens:
        return {f"BLEU-{i}": 0.0 for i in range(1, max_n + 1)}

    precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams = compute_ngram_counts(ref_tokens, n)
        cand_ngrams = compute_ngram_counts(cand_tokens, n)

        clipped_matches = sum(min(count, ref_ngrams[ng]) for ng, count in cand_ngrams.items())
        total_cand_ngrams = max(sum(cand_ngrams.values()), 1)

        p_n = clipped_matches / total_cand_ngrams
        precisions.append(p_n)

    # Brevity penalty
    c = len(cand_tokens)
    r = len(ref_tokens)
    bp = math.exp(1 - r / c) if c <= r else 1.0

    scores = {}
    for i in range(1, max_n + 1):
        # Geometric mean of precisions up to i
        sub_p = [p for p in precisions[:i]]
        if all(p > 0 for p in sub_p):
            geo_mean = math.exp(sum(math.log(p) for p in sub_p) / i)
            scores[f"BLEU-{i}"] = round(bp * geo_mean, 4)
        else:
            scores[f"BLEU-{i}"] = 0.0

    return scores


def compute_rouge_lcs(ref_tokens: List[str], cand_tokens: List[str]) -> int:
    """Longest common subsequence length."""
    m, n = len(ref_tokens), len(cand_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == cand_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def compute_rouge(reference: str, candidate: str) -> Dict[str, float]:
    """Computes ROUGE-1, ROUGE-2, and ROUGE-L F1 scores."""
    ref_tokens = re.findall(r"\w+", reference.lower())
    cand_tokens = re.findall(r"\w+", candidate.lower())

    if not cand_tokens or not ref_tokens:
        return {"ROUGE-1": 0.0, "ROUGE-2": 0.0, "ROUGE-L": 0.0}

    # ROUGE-1
    ref_1 = compute_ngram_counts(ref_tokens, 1)
    cand_1 = compute_ngram_counts(cand_tokens, 1)
    overlap_1 = sum(min(c, ref_1[ng]) for ng, c in cand_1.items())
    p_1 = overlap_1 / len(cand_tokens) if cand_tokens else 0.0
    r_1 = overlap_1 / len(ref_tokens) if ref_tokens else 0.0
    f1_1 = 2 * p_1 * r_1 / (p_1 + r_1) if (p_1 + r_1) > 0 else 0.0

    # ROUGE-2
    ref_2 = compute_ngram_counts(ref_tokens, 2)
    cand_2 = compute_ngram_counts(cand_tokens, 2)
    overlap_2 = sum(min(c, ref_2[ng]) for ng, c in cand_2.items())
    p_2 = overlap_2 / max(len(cand_tokens) - 1, 1)
    r_2 = overlap_2 / max(len(ref_tokens) - 1, 1)
    f1_2 = 2 * p_2 * r_2 / (p_2 + r_2) if (p_2 + r_2) > 0 else 0.0

    # ROUGE-L
    lcs = compute_rouge_lcs(ref_tokens, cand_tokens)
    p_l = lcs / len(cand_tokens) if cand_tokens else 0.0
    r_l = lcs / len(ref_tokens) if ref_tokens else 0.0
    f1_l = 2 * p_l * r_l / (p_l + r_l) if (p_l + r_l) > 0 else 0.0

    return {
        "ROUGE-1": round(f1_1, 4),
        "ROUGE-2": round(f1_2, 4),
        "ROUGE-L": round(f1_l, 4),
    }

