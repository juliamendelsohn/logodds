"""Log-odds ratio with a Dirichlet prior, using plain word-count dictionaries."""

from .core import compute_log_odds, log_odds_details, top_words

__version__ = "0.1.0"

__all__ = ["compute_log_odds", "log_odds_details", "top_words", "__version__"]
