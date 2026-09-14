# logodds

Log-odds ratio with a Dirichlet prior, for finding the words that distinguish
two corpora. Implementation of Monroe et al.'s "Fightin' Words" method, adapted from Jack Hessel's Python implementation.

This is a dependency-free implementation that works on **plain Python
dictionaries** mapping words to counts. It does not use scikit-learn, and there
are no vectorizers or arrays anywhere in the interface.

## Install

```bash
pip install logodds
```

## Usage

```python
from logodds import compute_log_odds, top_words

corpus1 = {"cat": 30, "purr": 12, "dog": 5, "the": 200}
corpus2 = {"cat": 4, "dog": 28, "leash": 9, "the": 210}

scores = compute_log_odds(corpus1, corpus2)

top_words(scores, n=20)             # words characteristic of corpus1
top_words(scores, n=20, corpus=2)   # words characteristic of corpus2
```

`compute_log_odds` returns a dict mapping each word to a z-score. Positive means
the word is characteristic of `corpus1`, negative means `corpus2`, and near zero
means it does not distinguish them.

### Informative prior

Pass counts from a larger background corpus:

```python
scores = compute_log_odds(corpus1, corpus2, prior_counts=background)
```

Each word's prior is its background count plus `alpha`, so α₀ is simply the sum
of those priors. Words that are common in the background get a large prior and
are shrunk hard toward zero.

Optional arguments:

| Argument | Default | |
| --- | --- | --- |
| `prior_counts` | `None` | Background corpus counts. Switches on the informative prior. |
| `alpha` | `0.01` | Without `prior_counts`, the pseudo-count added to every word. With it, a floor so no word gets a zero prior. |
| `min_count` | `0` | Drop words whose combined count is below this. |
| `vocabulary` | `None` | Restrict to a given word list. |

Corpus sizes are measured over the retained vocabulary, so filtering with
`min_count` or `vocabulary` renormalizes the frequencies.

### Intermediate values

`log_odds_details` takes the same arguments and returns `count1`, `count2`,
`freq1`, `freq2`, `prior`, `delta`, `variance` and `z` for each word.

## Method

For word *w* in corpus *i*, with count `y_i^w`, corpus size `n_i`, prior `α_w`
and `α_0 = Σ_w α_w`:

```
δ_i^w = log( (y_i^w + α_w) / (n_i + α_0 - y_i^w - α_w) )

δ^w = δ_1^w - δ_2^w

σ²(δ^w) ≈ 1/(y_1^w + α_w) + 1/(y_2^w + α_w)

z = δ^w / sqrt(σ²(δ^w))
```

> Monroe, B. L., Colaresi, M. P., & Quinn, K. M. (2008). Fightin' Words: Lexical
> Feature Selection and Evaluation for Identifying the Content of Political
> Conflict. *Political Analysis*, 16(4), 372–403.
> https://doi.org/10.1093/pan/mpn018

## Tests

```bash
pip install -e . pytest
python -m pytest -q
```

## License

MIT
