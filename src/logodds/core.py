"""Log-odds ratio with a Dirichlet prior (Monroe, Colaresi & Quinn 2008)."""

import math

__all__ = ["compute_log_odds", "log_odds_details", "top_words"]


def log_odds_details(
    counts1,
    counts2,
    prior_counts=None,
    alpha=0.01,
    min_count=0,
    vocabulary=None,
):
    """Same calculation as `compute_log_odds`, returning every intermediate value.

    Returns a dict mapping each word to a dict with keys `count1`, `count2`,
    `freq1`, `freq2`, `prior`, `delta` (the log-odds difference), `variance`
    (its estimated variance) and `z` (delta / sqrt(variance)).
    """
    if alpha <= 0:
        raise ValueError("alpha must be greater than 0.")

    vocab = set(counts1) | set(counts2) if vocabulary is None else set(vocabulary)
    if min_count > 0:
        vocab = {w for w in vocab if counts1.get(w, 0) + counts2.get(w, 0) >= min_count}
    if not vocab:
        return {}

    n1 = sum(counts1.get(w, 0) for w in vocab)
    n2 = sum(counts2.get(w, 0) for w in vocab)
    if n1 == 0 or n2 == 0:
        raise ValueError("Both corpora must contain at least one word in the vocabulary.")

    if prior_counts is None:
        alphas = {w: alpha for w in vocab}
    else:
        alphas = {w: prior_counts.get(w, 0) + alpha for w in vocab}
    alpha_0 = sum(alphas.values())

    results = {}
    for w in vocab:
        y1 = counts1.get(w, 0)
        y2 = counts2.get(w, 0)
        a_w = alphas[w]

        denom1 = n1 + alpha_0 - y1 - a_w
        denom2 = n2 + alpha_0 - y2 - a_w
        if denom1 <= 0 or denom2 <= 0:
            raise ValueError(
                f"Non-positive denominator for {w!r}: one word makes up nearly "
                "the whole corpus."
            )

        delta = math.log((y1 + a_w) / denom1) - math.log((y2 + a_w) / denom2)
        variance = 1.0 / (y1 + a_w) + 1.0 / (y2 + a_w)

        results[w] = {
            "count1": y1,
            "count2": y2,
            "freq1": y1 / n1,
            "freq2": y2 / n2,
            "prior": a_w,
            "delta": delta,
            "variance": variance,
            "z": delta / math.sqrt(variance),
        }

    return results


def compute_log_odds(
    counts1,
    counts2,
    prior_counts=None,
    alpha=0.01,
    min_count=0,
    vocabulary=None,
):
    """Return {word: z-score} for two corpora.

    A large positive score means the word is characteristic of `counts1`; a
    large negative score means it is characteristic of `counts2`.

    Parameters
    ----------
    counts1, counts2 : dict
        Word-count dicts. A word missing from one corpus counts as 0.
    prior_counts : dict, optional
        Word counts from a background corpus. Supplying this switches on the
        informative Dirichlet prior; leaving it None gives an uninformative
        (symmetric) prior.
    alpha : float, default 0.01
        Without `prior_counts`, the pseudo-count added to every word. With
        `prior_counts`, a floor so no word gets a zero prior.
    min_count : int, default 0
        Drop words whose combined count across both corpora is below this.
    vocabulary : iterable of str, optional
        Restrict to these words. Defaults to every word in either corpus.

    Corpus sizes are measured over the retained vocabulary, so filtering with
    `min_count` or `vocabulary` renormalizes the frequencies.

    Examples
    --------
    >>> corpus1 = {"cat": 30, "dog": 5, "the": 200}
    >>> corpus2 = {"cat": 4, "dog": 28, "the": 210}
    >>> scores = compute_log_odds(corpus1, corpus2)
    >>> scores["cat"] > 0 and scores["dog"] < 0
    True
    >>> abs(scores["the"]) < abs(scores["cat"])
    True
    """
    details = log_odds_details(
        counts1,
        counts2,
        prior_counts=prior_counts,
        alpha=alpha,
        min_count=min_count,
        vocabulary=vocabulary,
    )
    return {w: d["z"] for w, d in details.items()}


def top_words(scores, n=20, corpus=1):
    """Return the n highest-scoring words as (word, score) pairs.

    `corpus=1` gives the most positive scores (characteristic of the first
    corpus); `corpus=2` gives the most negative.

    Examples
    --------
    >>> top_words({"a": 3.0, "b": -2.0, "c": 0.1}, n=1)
    [('a', 3.0)]
    >>> top_words({"a": 3.0, "b": -2.0, "c": 0.1}, n=1, corpus=2)
    [('b', -2.0)]
    """
    if corpus not in (1, 2):
        raise ValueError("corpus must be 1 or 2.")
    ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=(corpus == 1))
    return ranked[:n]
