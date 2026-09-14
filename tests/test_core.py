"""Tests for the logodds package. Run with: python -m pytest -q"""

import math

import pytest

from logodds import compute_log_odds, log_odds_details, top_words

CORPUS_1 = {"the": 500, "cat": 40, "dog": 6, "purr": 9, "quantum": 1, "and": 300}
CORPUS_2 = {"the": 520, "cat": 5, "dog": 44, "bark": 11, "and": 290, "leash": 7}
BACKGROUND = {"the": 100000, "and": 80000, "cat": 50, "dog": 50, "purr": 5,
              "bark": 5, "leash": 5, "quantum": 5}


def test_direction_of_scores():
    scores = compute_log_odds(CORPUS_1, CORPUS_2)
    assert scores["cat"] > 0        # more characteristic of corpus 1
    assert scores["dog"] < 0        # more characteristic of corpus 2
    assert abs(scores["the"]) < abs(scores["cat"])   # shared word is near zero


def test_antisymmetry():
    a = compute_log_odds(CORPUS_1, CORPUS_2)
    b = compute_log_odds(CORPUS_2, CORPUS_1)
    for w in a:
        assert a[w] == pytest.approx(-b[w], rel=1e-12)


def test_missing_words_treated_as_zero():
    """A word absent from one corpus must not raise a KeyError."""
    scores = compute_log_odds({"a": 10, "b": 5}, {"a": 10})
    assert "b" in scores
    assert scores["b"] > 0


def test_rare_word_is_shrunk_toward_zero():
    """A 1-vs-0 word must score lower than a 100-vs-0 word."""
    c1 = {"common": 1000, "rare": 1, "frequent_marker": 100}
    c2 = {"common": 1000}
    scores = compute_log_odds(c1, c2)
    assert scores["frequent_marker"] > scores["rare"]


def test_larger_alpha_shrinks_rare_words_more():
    c1 = {"common": 1000, "rare": 2, "frequent_marker": 200}
    c2 = {"common": 1000, "other": 200}
    small = compute_log_odds(c1, c2, alpha=0.01)
    large = compute_log_odds(c1, c2, alpha=50.0)
    # The rare word loses more of its score than the frequent marker does.
    rare_shrink = large["rare"] / small["rare"]
    marker_shrink = large["frequent_marker"] / small["frequent_marker"]
    assert rare_shrink < marker_shrink


def test_informative_prior_shrinks_background_frequent_words():
    uninformative = compute_log_odds(CORPUS_1, CORPUS_2)
    informative = compute_log_odds(CORPUS_1, CORPUS_2, prior_counts=BACKGROUND)
    assert set(uninformative) == set(informative)
    # Function words are common in the background, so their prior is huge and
    # their scores are pulled hard toward zero.
    assert abs(informative["the"]) < abs(uninformative["the"])


def test_prior_is_background_count_plus_floor():
    background = {"the": 1000, "cat": 20}
    details = log_odds_details(CORPUS_1, CORPUS_2, prior_counts=background, alpha=0.01)
    assert details["the"]["prior"] == pytest.approx(1000.01)
    assert details["cat"]["prior"] == pytest.approx(20.01)
    assert details["quantum"]["prior"] == pytest.approx(0.01)


def test_prior_floor_prevents_zero_division():
    """A word absent from the background corpus must still get a score."""
    background = {"the": 1000}
    scores = compute_log_odds(CORPUS_1, CORPUS_2, prior_counts=background)
    assert math.isfinite(scores["quantum"])


def test_min_count_filters_vocabulary():
    scores = compute_log_odds(CORPUS_1, CORPUS_2, min_count=10)
    assert "quantum" not in scores   # total count 1
    assert "cat" in scores           # total count 45


def test_vocabulary_argument_restricts_output():
    scores = compute_log_odds(CORPUS_1, CORPUS_2, vocabulary=["cat", "dog"])
    assert set(scores) == {"cat", "dog"}


def test_details_fields():
    details = log_odds_details(CORPUS_1, CORPUS_2)
    d = details["cat"]
    assert d["count1"] == 40 and d["count2"] == 5
    assert d["freq1"] > d["freq2"]
    assert d["z"] == pytest.approx(d["delta"] / math.sqrt(d["variance"]))


def test_top_words():
    scores = compute_log_odds(CORPUS_1, CORPUS_2)
    top1 = top_words(scores, n=2, corpus=1)
    top2 = top_words(scores, n=2, corpus=2)
    assert top1[0][0] == "cat"
    assert top2[0][0] == "dog"
    assert top1[0][1] > top1[1][1]        # sorted descending
    assert top2[0][1] < top2[1][1]        # sorted ascending (most negative first)


def test_invalid_arguments():
    with pytest.raises(ValueError):
        compute_log_odds(CORPUS_1, CORPUS_2, alpha=0)
    with pytest.raises(ValueError):
        top_words({"a": 1.0}, corpus=3)
    with pytest.raises(ValueError):
        compute_log_odds({}, CORPUS_2)


def test_empty_vocabulary_returns_empty():
    assert log_odds_details(CORPUS_1, CORPUS_2, min_count=10**9) == {}


# ---------------------------------------------------------------------------
# Reference values produced by Jack Hessel's FightingWords implementation
# (https://github.com/jmhessel/FightingWords). To regenerate, call
# bayes_compare_language(l1, l2, prior=..., cv=...) on documents whose unigram
# counts equal the dicts below, passing an unfiltered CountVectorizer
# (token_pattern=r"(?u)\b\w+\b", so single-character words survive).
# ---------------------------------------------------------------------------
HESSEL_CORPUS_1 = {"a": 1, "and": 2, "at": 1, "cat": 5, "cats": 1, "dog": 1,
                   "door": 1, "ignores": 1, "mat": 1, "me": 1, "meow": 1,
                   "my": 1, "on": 1, "purr": 4, "sat": 1, "sleeps": 1,
                   "the": 6, "will": 1}
HESSEL_CORPUS_2 = {"a": 1, "adores": 1, "and": 2, "at": 1, "bark": 4, "cat": 1,
                   "dog": 5, "dogs": 1, "mailman": 1, "me": 1, "my": 1, "on": 1,
                   "rug": 1, "sat": 1, "sleeps": 1, "the": 6, "will": 1,
                   "woof": 1}
HESSEL_Z = {
    "a": 0.0,
    "adores": -0.4624826093074121,
    "and": 0.0,
    "at": 0.0,
    "bark": -0.6123390895316625,
    "cat": 1.598388995202337,
    "cats": 0.4624826093074121,
    "dog": -1.598388995202337,
    "dogs": -0.4624826093074121,
    "door": 0.4624826093074121,
    "ignores": 0.4624826093074121,
    "mailman": -0.4624826093074121,
    "mat": 0.4624826093074121,
    "me": 0.0,
    "meow": 0.4624826093074121,
    "my": 0.0,
    "on": 0.0,
    "purr": 0.6123390895316625,
    "rug": -0.4624826093074121,
    "sat": 0.0,
    "sleeps": 0.0,
    "the": 0.0,
    "will": 0.0,
    "woof": -0.4624826093074121,
}


def test_matches_hessel_fightingwords():
    actual = compute_log_odds(HESSEL_CORPUS_1, HESSEL_CORPUS_2, alpha=0.01)
    assert set(actual) == set(HESSEL_Z)
    for w, expected in HESSEL_Z.items():
        assert actual[w] == pytest.approx(expected, abs=1e-12)


# Same reference implementation, informative prior: the prior vector passed to
# bayes_compare_language is BACKGROUND[w] + 0.01 for each word in CORPUS_1/2.
HESSEL_INFORMATIVE_Z = {
    "and": 0.08664163039181341,
    "bark": -2.2693449734045288,
    "cat": 2.8790441554595216,
    "dog": -3.0686372149306269,
    "leash": -1.6437451631429068,
    "purr": 1.9757749829094806,
    "quantum": 0.3010232453335830,
    "the": -0.0418221428687848,
}


def test_matches_hessel_fightingwords_informative_prior():
    actual = compute_log_odds(CORPUS_1, CORPUS_2, prior_counts=BACKGROUND, alpha=0.01)
    assert set(actual) == set(HESSEL_INFORMATIVE_Z)
    for w, expected in HESSEL_INFORMATIVE_Z.items():
        assert actual[w] == pytest.approx(expected, abs=1e-12)
