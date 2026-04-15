import hypothesis
import numpy as np
import pandas as pd
from hypothesis import given, settings, assume
import pytest


@given(data=hypothesis.strategies.data())
@settings(max_examples=30)
def test_factorize_return_type(data):
    values = data.draw(
        hypothesis.strategies.lists(
            hypothesis.strategies.text(alphabet="abcdefghij", min_size=1, max_size=4),
            min_size=1,
            max_size=20,
        )
    )
    series = pd.Series(values)
    result = series.factorize()
    assert isinstance(result, tuple) and len(result) == 2


@given(data=hypothesis.strategies.data())
@settings(max_examples=30)
def test_codes_is_integer_ndarray(data):
    values = data.draw(
        hypothesis.strategies.lists(
            hypothesis.strategies.text(alphabet="abcdefghij", min_size=1, max_size=4),
            min_size=1,
            max_size=20,
        )
    )
    series = pd.Series(values)
    codes, uniques = series.factorize()
    assert isinstance(codes, np.ndarray) and np.issubdtype(codes.dtype, np.integer)


@given(data=hypothesis.strategies.data())
@settings(max_examples=30)
def test_codes_length_equals_series_length(data):
    values = data.draw(
        hypothesis.strategies.lists(
            hypothesis.strategies.text(alphabet="abcdefghij", min_size=1, max_size=4),
            min_size=1,
            max_size=20,
        ),
        label="values",
    )
    series = pd.Series(values)
    codes, uniques = series.factorize()
    assert len(codes) == len(series)


@given(data=hypothesis.strategies.data())
@settings(max_examples=30)
def test_reconstruction_correctness(data):
    values = data.draw(
        hypothesis.strategies.lists(
            hypothesis.strategies.text(alphabet="abcdefghij", min_size=1, max_size=4),
            min_size=1,
            max_size=20,
        )
    )
    series = pd.Series(values)
    codes, uniques = series.factorize()
    reconstructed = list(uniques.take(codes))
    assert reconstructed == list(series)


@given(data=hypothesis.strategies.data())
@settings(max_examples=30)
def test_uniques_has_no_duplicates(data):
    values = data.draw(
        hypothesis.strategies.lists(
            hypothesis.strategies.text(alphabet="abcdefghij", min_size=1, max_size=4),
            min_size=1,
            max_size=20,
        )
    )
    series = pd.Series(values)
    codes, uniques = series.factorize()
    assert len(uniques) == len(set(uniques))


@given(data=hypothesis.strategies.data())
@settings(max_examples=30)
def test_uniques_count_equals_distinct_values(data):
    values = data.draw(
        hypothesis.strategies.lists(
            hypothesis.strategies.text(alphabet="abcdefghij", min_size=1, max_size=4),
            min_size=1,
            max_size=20,
        )
    )
    series = pd.Series(values)
    codes, uniques = series.factorize()
    assert len(uniques) == series.dropna().nunique()


@given(data=hypothesis.strategies.data())
@settings(max_examples=30)
def test_uniques_type_is_index(data):
    values = data.draw(
        hypothesis.strategies.lists(
            hypothesis.strategies.text(alphabet="abcdefghij", min_size=1, max_size=4),
            min_size=1,
            max_size=20,
        )
    )
    series = pd.Series(values)
    codes, uniques = series.factorize()
    assert isinstance(uniques, pd.Index)


@given(data=hypothesis.strategies.data())
@settings(max_examples=30)
def test_sort_true_sorts_uniques(data):
    values = data.draw(
        hypothesis.strategies.lists(
            hypothesis.strategies.text(alphabet="abcdefghij", min_size=1, max_size=4),
            min_size=1,
            max_size=20,
        )
    )
    series = pd.Series(values)
    codes, uniques = series.factorize(sort=True)
    assert list(uniques) == sorted(list(uniques))


@given(data=hypothesis.strategies.data())
@settings(max_examples=30)
def test_sort_false_preserves_first_occurrence_order(data):
    values = data.draw(
        hypothesis.strategies.lists(
            hypothesis.strategies.text(alphabet="abcdefghij", min_size=1, max_size=4),
            min_size=1,
            max_size=20,
        )
    )
    series = pd.Series(values)
    codes, uniques = series.factorize(sort=False)
    expected_order = list(dict.fromkeys(v for v in series if not pd.isna(v)))
    assert list(uniques) == expected_order


@given(data=hypothesis.strategies.data())
@settings(max_examples=30)
def test_na_sentinel_true_uses_minus_one(data):
    values = data.draw(
        hypothesis.strategies.lists(
            hypothesis.strategies.one_of(
                hypothesis.strategies.text(alphabet="abcde", min_size=1, max_size=3),
                hypothesis.strategies.none(),
            ),
            min_size=2,
            max_size=20,
        ).filter(lambda x: any(v is None for v in x))
    )
    series = pd.Series(values)
    codes, uniques = series.factorize(use_na_sentinel=True)
    for i in range(len(series)):
        if pd.isna(series.iloc[i]):
            assert codes[i] == -1


@given(data=hypothesis.strategies.data())
@settings(max_examples=30)
def test_na_sentinel_true_non_na_codes_nonnegative(data):
    values = data.draw(
        hypothesis.strategies.lists(
            hypothesis.strategies.one_of(
                hypothesis.strategies.text(alphabet="abcde", min_size=1, max_size=3),
                hypothesis.strategies.none(),
            ),
            min_size=2,
            max_size=20,
        ).filter(lambda x: any(v is None for v in x))
    )
    series = pd.Series(values)
    codes, uniques = series.factorize(use_na_sentinel=True)
    for i in range(len(series)):
        if not pd.isna(series.iloc[i]):
            assert codes[i] >= 0


@given(data=hypothesis.strategies.data())
@settings(max_examples=30)
def test_na_sentinel_true_nan_not_in_uniques(data):
    values = data.draw(
        hypothesis.strategies.lists(
            hypothesis.strategies.one_of(
                hypothesis.strategies.text(alphabet="abcde", min_size=1, max_size=3),
                hypothesis.strategies.none(),
            ),
            min_size=2,
            max_size=20,
        ).filter(lambda x: any(v is None for v in x))
    )
    series = pd.Series(values)
    codes, uniques = series.factorize(use_na_sentinel=True)
    assert not any(pd.isna(v) for v in uniques)


@given(data=hypothesis.strategies.data())
@settings(max_examples=30)
def test_na_sentinel_false_all_codes_nonnegative(data):
    values = data.draw(
        hypothesis.strategies.lists(
            hypothesis.strategies.one_of(
                hypothesis.strategies.text(alphabet="abcde", min_size=1, max_size=3),
                hypothesis.strategies.none(),
            ),
            min_size=2,
            max_size=20,
        ).filter(lambda x: any(v is None for v in x))
    )
    series = pd.Series(values)
    codes, uniques = series.factorize(use_na_sentinel=False)
    assert all(c >= 0 for c in codes)


@given(data=hypothesis.strategies.data())
@settings(max_examples=30)
def test_na_sentinel_false_nan_in_uniques(data):
    values = data.draw(
        hypothesis.strategies.lists(
            hypothesis.strategies.one_of(
                hypothesis.strategies.text(alphabet="abcde", min_size=1, max_size=3),
                hypothesis.strategies.none(),
            ),
            min_size=2,
            max_size=20,
        ).filter(lambda x: any(v is None for v in x))
    )
    series = pd.Series(values)
    codes, uniques = series.factorize(use_na_sentinel=False)
    assert any(pd.isna(v) for v in uniques)


@given(data=hypothesis.strategies.data())
@settings(max_examples=30)
def test_all_same_series_has_one_unique(data):
    n = data.draw(hypothesis.strategies.integers(min_value=1, max_value=5))
    v = data.draw(hypothesis.strategies.integers(min_value=-100, max_value=100))
    series = pd.Series([v] * n)
    codes, uniques = series.factorize()
    assert len(uniques) == 1


@given(data=hypothesis.strategies.data())
@settings(max_examples=30)
def test_all_same_series_all_codes_zero(data):
    n = data.draw(hypothesis.strategies.integers(min_value=1, max_value=5))
    v = data.draw(hypothesis.strategies.integers(min_value=-100, max_value=100))
    series = pd.Series([v] * n)
    codes, uniques = series.factorize()
    assert all(c == 0 for c in codes)


def test_empty_series():
    series = pd.Series([], dtype=object)
    codes, uniques = series.factorize()
    assert len(codes) == 0 and len(uniques) == 0


@given(data=hypothesis.strategies.data())
@settings(max_examples=30)
def test_categorical_series_uniques_type(data):
    values = data.draw(
        hypothesis.strategies.lists(
            hypothesis.strategies.sampled_from(["a", "b", "c", "d"]),
            min_size=1,
            max_size=20,
        )
    )
    series = pd.Series(pd.Categorical(values))
    codes, uniques = series.factorize()
    assert isinstance(uniques, (pd.Categorical, pd.CategoricalIndex))
