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


# ============================================================================
# NEW TESTS BELOW - Cover non-happy-path cases from documentation
# ============================================================================


# Test with numeric Series (int) - documents that factorize works on any dtype
@given(data=hypothesis.strategies.data())
@settings(max_examples=30)
def test_factorize_numeric_int_series(data):
    values = data.draw(
        hypothesis.strategies.lists(
            hypothesis.strategies.integers(min_value=-100, max_value=100),
            min_size=1,
            max_size=20,
            unique=True,
        )
    )
    series = pd.Series(values)
    codes, uniques = series.factorize()
    assert len(codes) == len(series)
    assert len(uniques) == len(set(values))
    reconstructed = list(uniques.take(codes))
    assert reconstructed == list(series)


# Test with numeric Series (float) including NaN
@given(data=hypothesis.strategies.data())
@settings(max_examples=30)
def test_factorize_float_series_with_nan(data):
    values = data.draw(
        hypothesis.strategies.lists(
            hypothesis.strategies.one_of(
                hypothesis.strategies.floats(allow_nan=False, allow_infinity=False),
                hypothesis.strategies.none(),
            ),
            min_size=2,
            max_size=15,
        ).filter(lambda x: any(v is None for v in x))
    )
    series = pd.Series(values)
    codes, uniques = series.factorize(use_na_sentinel=True)
    assert all(codes[i] == -1 if pd.isna(values[i]) else codes[i] >= 0 for i in range(len(values)))
    assert not any(pd.isna(v) for v in uniques)


# Test sort=True with NaN values - documents interaction between sort and use_na_sentinel
@given(data=hypothesis.strategies.data())
@settings(max_examples=30)
def test_sort_true_with_na_values(data):
    values = data.draw(
        hypothesis.strategies.lists(
            hypothesis.strategies.one_of(
                hypothesis.strategies.text(alphabet="abc", min_size=1, max_size=2),
                hypothesis.strategies.none(),
            ),
            min_size=3,
            max_size=15,
        ).filter(lambda x: any(v is None for v in x))
    )
    series = pd.Series(values)
    codes, uniques = series.factorize(sort=True, use_na_sentinel=True)
    non_none_values = [v for v in values if v is not None]
    unique_non_none = list(dict.fromkeys(non_none_values))
    assert list(uniques) == sorted(unique_non_none)
    assert all(codes[i] == -1 if pd.isna(values[i]) else codes[i] >= 0 for i in range(len(values)))


# Test reconstruction works correctly with sort=True
@given(data=hypothesis.strategies.data())
@settings(max_examples=30)
def test_reconstruction_with_sort_true(data):
    values = data.draw(
        hypothesis.strategies.lists(
            hypothesis.strategies.text(alphabet="abcdef", min_size=1, max_size=4),
            min_size=1,
            max_size=15,
        )
    )
    series = pd.Series(values)
    codes, uniques = series.factorize(sort=True)
    reconstructed = list(uniques.take(codes))
    assert reconstructed == list(series)


# Test Categorical preserves categories not in values (from docs)
def test_categorical_preserves_unseen_categories():
    cat = pd.Categorical(["a", "a", "c"], categories=["a", "b", "c"])
    codes, uniques = pd.factorize(cat)
    assert list(codes) == [0, 0, 1]
    assert list(uniques.categories) == ["a", "b", "c"]
    assert set(uniques.categories) == {"a", "b", "c"}


# Test with use_na_sentinel=False includes NaN in uniques (from docs example)
def test_use_na_sentinel_false_includes_nan_in_uniques():
    values = np.array([1, 2, 1, np.nan])
    codes, uniques = pd.factorize(values, use_na_sentinel=False)
    assert list(codes) == [0, 1, 0, 2]
    assert len(uniques) == 3
    assert uniques[0] == 1.0
    assert uniques[1] == 2.0
    assert np.isnan(uniques[2])


# Test with use_na_sentinel=True (default) uses -1 for NaN (from docs example)
def test_use_na_sentinel_true_uses_minus_one_for_nan():
    values = np.array([1, 2, 1, np.nan])
    codes, uniques = pd.factorize(values, use_na_sentinel=True)
    assert list(codes) == [0, 1, 0, -1]
    assert list(uniques) == [1.0, 2.0]


# Test with empty series and sort=True - edge case
def test_empty_series_with_sort_true():
    series = pd.Series([], dtype=object)
    codes, uniques = series.factorize(sort=True)
    assert len(codes) == 0 and len(uniques) == 0


# Test with empty series and use_na_sentinel=False - edge case
def test_empty_series_with_na_sentinel_false():
    series = pd.Series([], dtype=float)
    codes, uniques = series.factorize(use_na_sentinel=False)
    assert len(codes) == 0 and len(uniques) == 0


# Test that codes dtype is appropriate for the size of uniques
@given(data=hypothesis.strategies.data())
@settings(max_examples=20)
def test_codes_dtype_matches_uniques_size(data):
    n_uniques = data.draw(hypothesis.strategies.integers(min_value=1, max_value=100))
    values = list(range(n_uniques)) * 2
    series = pd.Series(values)
    codes, uniques = series.factorize()
    assert codes.max() == n_uniques - 1


# Test all NaN series - edge case
def test_all_nan_series():
    series = pd.Series([None, None, None])
    codes, uniques = series.factorize(use_na_sentinel=True)
    assert all(c == -1 for c in codes)
    assert len(uniques) == 0


# Test all NaN series with use_na_sentinel=False
def test_all_nan_series_with_na_sentinel_false():
    series = pd.Series([None, None, None])
    codes, uniques = series.factorize(use_na_sentinel=False)
    assert len(codes) == 3
    assert len(uniques) == 1
    assert pd.isna(uniques[0])


# Test with boolean Series
def test_boolean_series():
    series = pd.Series([True, False, True, False, True])
    codes, uniques = series.factorize()
    assert len(uniques) == 2
    reconstructed = list(uniques.take(codes))
    assert reconstructed == list(series)


# Test with sort=True on boolean Series
def test_boolean_series_sort_true():
    series = pd.Series([True, False, True, False, True])
    codes, uniques = series.factorize(sort=True)
    assert list(uniques) == [False, True]


# Test reconstruction works with use_na_sentinel=False
@given(data=hypothesis.strategies.data())
@settings(max_examples=20)
def test_reconstruction_with_na_sentinel_false(data):
    values = data.draw(
        hypothesis.strategies.lists(
            hypothesis.strategies.one_of(
                hypothesis.strategies.integers(min_value=0, max_value=50),
                hypothesis.strategies.none(),
            ),
            min_size=2,
            max_size=15,
        ).filter(lambda x: any(v is None for v in x))
    )
    series = pd.Series(values)
    codes, uniques = series.factorize(use_na_sentinel=False)
    reconstructed = []
    for c in codes:
        if c >= 0:
            reconstructed.append(uniques[c])
        else:
            reconstructed.append(np.nan)
    assert all(pd.isna(reconstructed[i]) if pd.isna(values[i]) else reconstructed[i] == values[i]
               for i in range(len(values)))


# Test single element series
def test_single_element_series():
    series = pd.Series(["a"])
    codes, uniques = series.factorize()
    assert len(codes) == 1
    assert len(uniques) == 1
    assert codes[0] == 0
    assert uniques[0] == "a"


# Test single element series with None
def test_single_none_element_series():
    series = pd.Series([None])
    codes, uniques = series.factorize(use_na_sentinel=True)
    assert codes[0] == -1
    assert len(uniques) == 0
