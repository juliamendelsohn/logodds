# logodds

Log-odds ratio with a Dirichlet prior, for finding the words that distinguish
two corpora.

This is a dependency-free implementation that works on **plain Python
dictionaries** mapping words to counts. It does not use scikit-learn, and there
are no vectorizers or arrays anywhere in the interface.

It extends a compact version written by Julia Mendelsohn for INST425 (AI for
Text Analysis) at the University of Maryland, adding support for the
informative prior.

Output is verified against [Jack Hessel's FightingWords](https://github.com/jmhessel/FightingWords),
the reference implementation of this method; the test suite checks the z-scores
match exactly.

## Install

```bash
pip install git+https://github.com/juliamendelsohn/logodds.git
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

Optional arguments:

| Argument | Default | |
| --- | --- | --- |
| `prior_counts` | `None` | Background corpus counts. Switches on the informative prior. |
| `alpha` | `0.01` | Without `prior_counts`, the pseudo-count added to every word. With it, a floor so no word gets a zero prior. |
| `prior_strength` | `None` | Total prior mass (α₀) to spread over the vocabulary, proportional to the background counts. `None` uses the raw counts. |
| `min_count` | `0` | Drop words whose combined count is below this. |
| `vocabulary` | `None` | Restrict to a given word list. |

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
python -m pytest tests/ -q
```

## License

MIT
