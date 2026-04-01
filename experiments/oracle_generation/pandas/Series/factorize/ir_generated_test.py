import numpy as np
import pandas as pd
from hypothesis import given, strategies as st


def _first_occurrence_uniques(values):
    uniques = []
    for value in values:
        if pd.isna(value):
            continue
        if not any(existing == value for existing in uniques):
            uniques.append(value)
    return uniques


def _assert_same_values(left, right):
    assert len(left) == len(right)
    for observed, expected in zip(left, right):
        if pd.isna(expected):
            assert pd.isna(observed)
        else:
            assert observed == expected


small_text = st.text(alphabet="abc", min_size=1, max_size=3)
non_missing_values = st.lists(small_text, min_size=1, max_size=8)
with_missing_values = st.lists(
    st.one_of(small_text, st.none(), st.just(np.nan)),
    min_size=1,
    max_size=8,
).filter(lambda xs: any(pd.isna(x) for x in xs))


@given(non_missing_values)
def test_factorize_reconstructs_original_for_plain_series(values):
    s = pd.Series(values, dtype=object)

    codes, uniques = s.factorize(sort=False, use_na_sentinel=True)

    assert isinstance(codes, np.ndarray)
    assert np.issubdtype(codes.dtype, np.integer)
    assert isinstance(uniques, pd.Index)
    assert len(codes) == len(s)
    _assert_same_values(uniques.take(codes).tolist(), s.tolist())


@given(non_missing_values)
def test_factorize_preserves_first_occurrence_order_when_sort_false(values):
    s = pd.Series(values, dtype=object)

    _, uniques = s.factorize(sort=False, use_na_sentinel=True)

    assert uniques.tolist() == _first_occurrence_uniques(s.tolist())


@given(non_missing_values)
def test_factorize_sort_true_matches_sort_false_after_relabel(values):
    s = pd.Series(values, dtype=object)

    codes_unsorted, uniques_unsorted = s.factorize(sort=False, use_na_sentinel=True)
    codes_sorted, uniques_sorted = s.factorize(sort=True, use_na_sentinel=True)

    assert uniques_sorted.tolist() == sorted(uniques_sorted.tolist())
    _assert_same_values(uniques_unsorted.take(codes_unsorted).tolist(), uniques_sorted.take(codes_sorted).tolist())


@given(with_missing_values)
def test_factorize_use_na_sentinel_true_excludes_missing_from_uniques(values):
    s = pd.Series(values, dtype=object)

    codes, uniques = s.factorize(sort=False, use_na_sentinel=True)

    assert len(codes) == len(s)
    assert not any(pd.isna(value) for value in uniques.tolist())
    for code, value in zip(codes.tolist(), s.tolist()):
        if pd.isna(value):
            assert code == -1
        else:
            assert code >= 0
            assert uniques[code] == value


@given(with_missing_values)
def test_factorize_use_na_sentinel_false_round_trips_and_keeps_missing(values):
    s = pd.Series(values, dtype=object)

    codes, uniques = s.factorize(sort=False, use_na_sentinel=False)

    assert all(code >= 0 for code in codes.tolist())
    assert any(pd.isna(value) for value in uniques.tolist())
    _assert_same_values(uniques.take(codes).tolist(), s.tolist())


@given(with_missing_values)
def test_factorize_missing_variants_share_one_code_when_sentinel_false(values):
    s = pd.Series(values, dtype=object)

    codes, uniques = s.factorize(sort=False, use_na_sentinel=False)

    missing_codes = {code for code, value in zip(codes.tolist(), s.tolist()) if pd.isna(value)}
    assert len(missing_codes) == 1
    missing_code = next(iter(missing_codes))
    assert pd.isna(uniques[missing_code])


@given(st.lists(st.integers(min_value=-2, max_value=2), min_size=1, max_size=6))
def test_factorize_all_same_values_collapse_to_zero_code(values):
    repeated = pd.Series([values[0]] * len(values))

    codes, uniques = repeated.factorize(sort=False, use_na_sentinel=True)

    assert uniques.tolist() == [values[0]]
    assert codes.tolist() == [0] * len(repeated)


def test_factorize_empty_series_returns_empty_outputs():
    s = pd.Series([], dtype=object)

    codes, uniques = s.factorize(sort=False, use_na_sentinel=True)

    assert isinstance(codes, np.ndarray)
    assert len(codes) == 0
    assert len(uniques) == 0


@given(st.lists(st.sampled_from(["a", "b", "c"]), min_size=1, max_size=8))
def test_factorize_categorical_series_returns_categorical_uniques(values):
    s = pd.Series(pd.Categorical(values, categories=["c", "b", "a"], ordered=True))

    codes, uniques = s.factorize(sort=False, use_na_sentinel=True)

    assert isinstance(uniques, (pd.Categorical, pd.CategoricalIndex))
    _assert_same_values(list(uniques.take(codes)), s.tolist())


@given(with_missing_values)
def test_factorize_sort_flag_does_not_change_missing_mask(values):
    s = pd.Series(values, dtype=object)

    codes_unsorted, uniques_unsorted = s.factorize(sort=False, use_na_sentinel=True)
    codes_sorted, uniques_sorted = s.factorize(sort=True, use_na_sentinel=True)

    assert [code == -1 for code in codes_unsorted.tolist()] == [pd.isna(v) for v in s.tolist()]
    assert [code == -1 for code in codes_sorted.tolist()] == [pd.isna(v) for v in s.tolist()]
    _assert_same_values(
        [uniques_unsorted[code] if code >= 0 else np.nan for code in codes_unsorted.tolist()],
        [uniques_sorted[code] if code >= 0 else np.nan for code in codes_sorted.tolist()],
    )


@given(st.lists(st.sampled_from(["a", "b", "c"]), min_size=1, max_size=8))
def test_factorize_categorical_sort_true_round_trips(values):
    s = pd.Series(pd.Categorical(values, categories=["c", "b", "a"], ordered=True))

    codes, uniques = s.factorize(sort=True, use_na_sentinel=True)

    assert isinstance(uniques, (pd.Categorical, pd.CategoricalIndex))
    _assert_same_values(list(uniques.take(codes)), s.tolist())
