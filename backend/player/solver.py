"""
Information-theoretic Wordle solver.

Core algorithm:
  For each candidate guess word, compute the expected entropy reduction
  over the remaining candidate set. The word with the highest entropy
  (most bits of information) is the optimal next guess.

  Entropy for a guess G against remaining candidates C:
      E[G] = -Σ p_k * log2(p_k)
  where p_k = |{c ∈ C : pattern(G,c) == k}| / |C|
  and k ranges over all 3^5 = 243 possible color patterns.

Also computes per-position saliency scores to explain why a word was suggested.
"""

import math
from collections import Counter
from functools import lru_cache

from .wordlist import get_word_list, filter_by_constraints, parse_board_to_constraints


# ---------------------------------------------------------------------------
# Pattern computation
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def compute_pattern(guess: str, answer: str) -> int:
    """
    Compute the Wordle color pattern as an integer in [0, 243).

    Encoding per position:
        0 = absent  (gray)
        1 = present (yellow)
        2 = correct (green)

    Pattern integer = sum(p[i] * 3^i for i in range(5))

    Handles duplicate letters correctly using the two-pass method:
    1. First mark correct (green) positions
    2. Then mark present (yellow) using remaining unmatched letters in answer
    """
    pattern = [0] * 5
    answer_remaining = list(answer)

    # Pass 1: greens
    for i in range(5):
        if guess[i] == answer[i]:
            pattern[i] = 2
            answer_remaining[i] = None  # consumed

    # Pass 2: yellows
    for i in range(5):
        if pattern[i] == 2:
            continue
        if guess[i] in answer_remaining:
            pattern[i] = 1
            answer_remaining[answer_remaining.index(guess[i])] = None

    return sum(pattern[i] * (3 ** i) for i in range(5))


def pattern_to_result(pattern_int: int) -> list[str]:
    """Convert a pattern integer to a list of 5 state strings."""
    STATE_MAP = {0: "absent", 1: "present", 2: "correct"}
    result = []
    remaining = pattern_int
    for _ in range(5):
        result.append(STATE_MAP[remaining % 3])
        remaining //= 3
    return result


# ---------------------------------------------------------------------------
# Entropy engine
# ---------------------------------------------------------------------------

def compute_entropy(guess: str, candidates: list[str]) -> float:
    """
    Expected information gain (in bits) of guessing `guess`
    against the remaining `candidates`.
    """
    if not candidates:
        return 0.0

    counts = Counter(compute_pattern(guess, ans) for ans in candidates)
    n = len(candidates)
    entropy = 0.0
    for count in counts.values():
        p = count / n
        entropy -= p * math.log2(p)

    return entropy


def _letter_freq_score(word: str, candidates: list[str]) -> float:
    """
    Fast heuristic: score a word by letter frequency across remaining candidates.
    Used to pre-rank large candidate sets before full entropy calculation.
    """
    from collections import Counter
    # Count unique letters at each position across all candidates
    pos_freq: list[Counter] = [Counter() for _ in range(5)]
    for cand in candidates:
        seen = set()
        for i, ch in enumerate(cand):
            pos_freq[i][ch] += 1
            seen.add(ch)

    score = 0.0
    seen_in_word = set()
    for i, ch in enumerate(word):
        if ch not in seen_in_word:  # reward unique letters
            score += pos_freq[i].get(ch, 0)
            seen_in_word.add(ch)
    return score


# ---------------------------------------------------------------------------
# Position saliency
# ---------------------------------------------------------------------------

def compute_position_saliency(
    word: str,
    candidates: list[str],
    baseline_entropy: float,
) -> list[float]:
    """
    Per-position saliency: how much does each letter contribute to the
    word's entropy score?

    Method: counterfactual ablation.
    For position i, measure the average entropy over all 26 possible
    substitute letters. The saliency is how much worse (lower entropy)
    the word would be on average if position i had a random letter instead.

    Returns:
        list of 5 floats in [0, 1], normalized so max = 1.0
    """
    if baseline_entropy < 1e-9 or not candidates:
        return [0.0] * 5

    alphabet = "abcdefghijklmnopqrstuvwxyz"
    saliency = []

    for pos in range(5):
        counterfactual_entropies = []
        for sub in alphabet:
            if sub == word[pos]:
                continue
            alt_word = word[:pos] + sub + word[pos + 1:]
            # Only consider if it's a plausible substitution (not necessarily valid)
            e = compute_entropy(alt_word, candidates)
            counterfactual_entropies.append(e)

        if not counterfactual_entropies:
            saliency.append(0.0)
            continue

        avg_alt = sum(counterfactual_entropies) / len(counterfactual_entropies)
        # How much better is the actual letter vs. a random one?
        contribution = baseline_entropy - avg_alt
        saliency.append(max(0.0, contribution))

    # Normalize to [0, 1]
    max_sal = max(saliency) if saliency else 1.0
    if max_sal > 1e-9:
        saliency = [s / max_sal for s in saliency]

    return saliency


def compute_letter_contributions(
    word: str,
    candidates: list[str],
) -> dict[str, float]:
    """
    For each unique letter in the word, estimate how much it individually
    reduces the remaining candidate pool.

    Returns a dict {letter: normalized_contribution}.
    """
    if not candidates:
        return {ch: 0.0 for ch in word}

    n_total = len(candidates)
    contributions: dict[str, float] = {}

    for ch in set(word):
        # Words containing this letter (as present constraint)
        with_letter = [c for c in candidates if ch in c]
        without_letter = [c for c in candidates if ch not in c]
        # Reduction = fraction of candidates eliminated if letter is absent
        reduction = len(without_letter) / n_total
        contributions[ch] = reduction

    # Normalize to sum to 1
    total = sum(contributions.values())
    if total > 1e-9:
        contributions = {k: v / total for k, v in contributions.items()}

    return contributions


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

class WordleSolver:
    """
    Information-theoretic Wordle solver.

    Usage:
        solver = WordleSolver()
        results = solver.get_suggestions(board_rows, n=5)
    """

    def __init__(self):
        self._all_words = get_word_list()
        # Compute and cache top openers by entropy at init time.
        # We evaluate all 2315 words against the full candidate set once.
        print("[Solver] Pre-computing opener entropy scores…")
        scored = sorted(
            ((w, compute_entropy(w, self._all_words)) for w in self._all_words),
            key=lambda x: x[1],
            reverse=True,
        )
        self._openers = scored[:20]  # top-20 cached as (word, entropy) pairs
        print(f"[Solver] Best opener: {self._openers[0][0].upper()} "
              f"({self._openers[0][1]:.3f} bits)")

    def get_suggestions(
        self,
        board_rows: list[list[dict]],
        n: int = 5,
    ) -> dict:
        """
        Compute top-n word suggestions given the current board state.

        Args:
            board_rows: list of rows from /api/analyze (or manually constructed)
            n: number of suggestions to return

        Returns:
            {
              "suggestions": [...],
              "candidates_remaining": int,
              "stats": {...},
            }
        """
        green, yellow, gray, must_contain = parse_board_to_constraints(board_rows)
        candidates = filter_by_constraints(
            self._all_words, green, yellow, gray, must_contain
        )

        # No guesses yet → return openers
        total_filled = sum(
            1
            for row in board_rows
            for tile in row
            if tile.get("state", "empty") != "empty"
        )

        if total_filled == 0:
            suggestions = []
            for opener, e in self._openers[:n]:
                sal = compute_position_saliency(opener, self._all_words, e)
                lc = compute_letter_contributions(opener, self._all_words)
                # Worst-case remaining words if this opener is guessed
                best_pattern_count = max(
                    Counter(compute_pattern(opener, c) for c in self._all_words).values()
                )
                suggestions.append({
                    "word": opener.upper(),
                    "entropy": round(e, 3),
                    "remaining_words": best_pattern_count,
                    "position_saliency": [round(s, 3) for s in sal],
                    "letter_contributions": {k.upper(): round(v, 3) for k, v in lc.items()},
                })
            return {
                "suggestions": suggestions,
                "candidates_remaining": len(self._all_words),
                "stats": {"opener_mode": True},
            }

        if len(candidates) == 0:
            return {
                "suggestions": [],
                "candidates_remaining": 0,
                "stats": {"error": "no_candidates"},
            }

        if len(candidates) <= n:
            # Return remaining candidates directly
            suggestions = []
            for word in candidates[:n]:
                e = compute_entropy(word, candidates)
                sal = compute_position_saliency(word, candidates, e)
                lc = compute_letter_contributions(word, candidates)
                suggestions.append({
                    "word": word.upper(),
                    "entropy": round(e, 3),
                    "remaining_words": 1,
                    "position_saliency": [round(s, 3) for s in sal],
                    "letter_contributions": {k.upper(): round(v, 3) for k, v in lc.items()},
                })
            return {
                "suggestions": suggestions,
                "candidates_remaining": len(candidates),
                "stats": {"exact_match": True},
            }

        # Score all candidates (or a fast-heuristic pre-filtered subset for very large sets)
        if len(candidates) > 1000:
            # Only use heuristic pre-filtering for very large sets (early game).
            # For <=1000 candidates, full entropy is fast enough (~50ms).
            scored = sorted(
                candidates,
                key=lambda w: _letter_freq_score(w, candidates),
                reverse=True,
            )
            eval_set = scored[:600]
        else:
            eval_set = candidates

        scored_with_entropy = [
            (w, compute_entropy(w, candidates))
            for w in eval_set
        ]
        scored_with_entropy.sort(key=lambda x: x[1], reverse=True)

        suggestions = []
        for word, entropy_val in scored_with_entropy[:n]:
            sal = compute_position_saliency(word, candidates, entropy_val)
            lc = compute_letter_contributions(word, candidates)

            # Estimate remaining words if this guess is optimal
            best_pattern_count = max(
                Counter(compute_pattern(word, c) for c in candidates).values()
            )

            suggestions.append({
                "word": word.upper(),
                "entropy": round(entropy_val, 3),
                "remaining_words": best_pattern_count,
                "position_saliency": [round(s, 3) for s in sal],
                "letter_contributions": {k.upper(): round(v, 3) for k, v in lc.items()},
            })

        return {
            "suggestions": suggestions,
            "candidates_remaining": len(candidates),
            "stats": {
                "total_words": len(self._all_words),
                "filtered": len(self._all_words) - len(candidates),
            },
        }
