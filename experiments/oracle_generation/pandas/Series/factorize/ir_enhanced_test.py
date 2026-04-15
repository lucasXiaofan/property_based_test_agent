import hypothesis
import numpy as np
import pandas as pd
from hypothesis import given, settings, assume
import pytest


# ============================================================================
# BASELINE TESTS (from baseline_test.py)
# ============================================================================


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


# ============================================================================
# NEW TESTS (inspired by IR - edge cases beyond baseline)
# ============================================================================


# NEW (IR): Test numeric float series - baseline only tested strings
@given(data=hypothesis.strategies.data())
@settings(max_examples=30)
def test_numeric_float_series_factorize(data):
    values = data.draw(
        hypothesis.strategies.lists(
            hypothesis.strategies.floats(
                allow_nan=False, allow_infinity=False, min_value=-100.0, max_value=100.0
            ),
            min_size=1,
            max_size=20,
        )
    )
    series = pd.Series(values)
    codes, uniques = series.factorize()
    assert len(codes) == len(series)
    assert len(uniques) == series.nunique()


# NEW (IR): Test reconstruction with numeric floats (high-stakes: precision issues)
@given(data=hypothesis.strategies.data())
@settings(max_examples=30)
def test_numeric_reconstruction_correctness(data):
    values = data.draw(
        hypothesis.strategies.lists(
            hypothesis.strategies.floats(
                allow_nan=False, allow_infinity=False, min_value=-100.0, max_value=100.0
            ),
            min_size=1,
            max_size=20,
        )
    )
    series = pd.Series(values)
    codes, uniques = series.factorize()
    reconstructed = uniques.take(codes)
    for i in range(len(series)):
        assert reconstructed[i] == series.iloc[i]


# NEW (IR): Test sort=True with NaN values - how does sorting interact with NaN sentinel?
@given(data=hypothesis.strategies.data())
@settings(max_examples=30)
def test_sort_true_with_nan_use_na_sentinel_true(data):
    values = data.draw(
        hypothesis.strategies.lists(
            hypothesis.strategies.one_of(
                hypothesis.strategies.text(alphabet="abcde", min_size=1, max_size=3),
                hypothesis.strategies.none(),
            ),
            min_size=3,
            max_size=15,
        ).filter(lambda x: any(v is None for v in x))
    )
    series = pd.Series(values)
    codes, uniques = series.factorize(sort=True, use_na_sentinel=True)
    assert list(uniques) == sorted([v for v in uniques if not pd.isna(v)])
    assert not any(pd.isna(v) for v in uniques)


# NEW (IR): Test sort=True with use_na_sentinel=False - combined edge case
@given(data=hypothesis.strategies.data())
@settings(max_examples=30)
def test_sort_true_with_na_sentinel_false(data):
    values = data.draw(
        hypothesis.strategies.lists(
            hypothesis.strategies.one_of(
                hypothesis.strategies.text(alphabet="abcde", min_size=1, max_size=3),
                hypothesis.strategies.none(),
            ),
            min_size=3,
            max_size=15,
        ).filter(lambda x: any(v is None for v in x))
    )
    series = pd.Series(values)
    codes, uniques = series.factorize(sort=True, use_na_sentinel=False)
    non_nan_uniques = [v for v in uniques if not pd.isna(v)]
    assert non_nan_uniques == sorted(non_nan_uniques)
    assert any(pd.isna(v) for v in uniques)


# NEW (IR): Reconstruction correctness with NaN when use_na_sentinel=False
@given(data=hypothesis.strategies.data())
@settings(max_examples=30)
def test_reconstruction_with_nan_sentinel_false(data):
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
    reconstructed = uniques.take(codes)
    for i in range(len(series)):
        orig_val = series.iloc[i]
        recon_val = reconstructed[i]
        if pd.isna(orig_val):
            assert pd.isna(recon_val)
        else:
            assert recon_val == orig_val


# NEW (IR): Empty string values in series - edge case for string handling
@given(data=hypothesis.strategies.data())
@settings(max_examples=30)
def test_empty_string_values(data):
    values = data.draw(
        hypothesis.strategies.lists(
            hypothesis.strategies.text(alphabet="abcde", min_size=0, max_size=3),
            min_size=1,
            max_size=20,
        ).filter(lambda x: "" in x)
    )
    series = pd.Series(values)
    codes, uniques = series.factorize()
    assert len(codes) == len(series)
    assert "" in uniques


# NEW (IR): Test codes are non-negative for non-NaN values when sort=True
@given(data=hypothesis.strategies.data())
@settings(max_examples=30)
def test_sort_true_non_nan_codes_nonnegative(data):
    values = data.draw(
        hypothesis.strategies.lists(
            hypothesis.strategies.one_of(
                hypothesis.strategies.text(alphabet="abcde", min_size=1, max_size=3),
                hypothesis.strategies.none(),
            ),
            min_size=3,
            max_size=15,
        ).filter(lambda x: any(v is None for v in x))
    )
    series = pd.Series(values)
    codes, uniques = series.factorize(sort=True, use_na_sentinel=True)
    for i in range(len(series)):
        if not pd.isna(series.iloc[i]):
            assert codes[i] >= 0


# NEW (IR): Verify uniques length equals number of unique non-NaN values in original
@given(data=hypothesis.strategies.data())
@settings(max_examples=30)
def test_uniques_count_with_nan_sentinel_true(data):
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
    unique_non_nan_count = series.dropna().nunique()
    assert len(uniques) == unique_non_nan_count


# NEW (IR): Test with all NaN values - edge case
def test_all_nan_series():
    series = pd.Series([None, None, None])
    codes, uniques = series.factorize(use_na_sentinel=True)
    assert len(uniques) == 0
    assert all(c == -1 for c in codes)


# NEW (IR): Test with all NaN and use_na_sentinel=False
def test_all_nan_series_sentinel_false():
    series = pd.Series([None, None, None])
    codes, uniques = series.factorize(use_na_sentinel=False)
    assert len(uniques) == 1
    assert pd.isna(uniques[0])
    assert all(c >= 0 for c in codes)


# NEW (IR): Codes values are valid indices into uniques
@given(data=hypothesis.strategies.data())
@settings(max_examples=30)
def test_codes_are_valid_indices(data):
    values = data.draw(
        hypothesis.strategies.lists(
            hypothesis.strategies.text(alphabet="abcdefghij", min_size=1, max_size=4),
            min_size=1,
            max_size=20,
        )
    )
    series = pd.Series(values)
    codes, uniques = series.factorize()
    assert all(0 <= c < len(uniques) for c in codes)
