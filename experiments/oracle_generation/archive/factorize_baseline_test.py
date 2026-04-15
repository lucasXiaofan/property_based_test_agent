"""
Baseline property-based tests for pandas.Series.factorize.

Properties tested:
1.  Return type: codes is a numpy ndarray of integers; uniques is a pandas Index for Series input
2.  Reconstruction: uniques.take(codes) equals original values for non-NaN entries
3.  codes are valid indices into uniques (all non-sentinel codes in [0, len(uniques)))
4.  Sentinel -1 assigned to NaN when use_na_sentinel=True (default)
5.  No NaN in uniques when use_na_sentinel=True
6.  NaN gets a non-negative code when use_na_sentinel=False; NaN appears in uniques
7.  len(uniques) equals number of distinct non-NaN values (use_na_sentinel=True)
8.  Each distinct value maps to exactly one code (consistency)
9.  sort=False: uniques appear in first-occurrence order
10. sort=True: uniques are sorted
11. sort=True + sort=False produce equivalent factorizations (same uniques set, consistent codes)
12. Empty Series: codes is empty array, uniques is empty Index
13. All-NaN Series with use_na_sentinel=True: all codes are -1, uniques is empty
14. All-NaN Series with use_na_sentinel=False: all codes are non-negative, NaN in uniques
15. Single-element Series: one unique, code is 0
16. Repeated single value: all codes identical
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.pandas import indexes, range_indexes


# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------

scalar_elements = st.one_of(
    st.integers(-50, 50),
    st.text(alphabet="abcde", min_size=1, max_size=3),
)

nullable_scalars = st.one_of(
    st.integers(-20, 20),
    st.text(alphabet="abcde", min_size=1, max_size=2),
    st.none(),
)


def series_of(elements, min_size=1, max_size=10):
    return st.lists(elements, min_size=min_size, max_size=max_size).map(pd.Series)


# ---------------------------------------------------------------------------
# 1. Return types
# ---------------------------------------------------------------------------

@given(s=series_of(scalar_elements))
@settings(max_examples=100)
def test_return_types(s):
    codes, uniques = s.factorize()
    assert isinstance(codes, np.ndarray)
    assert np.issubdtype(codes.dtype, np.integer)
    assert isinstance(uniques, pd.Index)


# ---------------------------------------------------------------------------
# 2. Reconstruction: uniques.take(codes) equals original (no NaN input)
# ---------------------------------------------------------------------------

@given(s=series_of(scalar_elements))
@settings(max_examples=150)
def test_reconstruction_no_nan(s):
    codes, uniques = s.factorize()
    reconstructed = uniques.take(codes)
    pd.testing.assert_index_equal(reconstructed, pd.Index(s))


# ---------------------------------------------------------------------------
# 3. codes are valid indices into uniques (non-sentinel codes in [0, len(uniques)))
# ---------------------------------------------------------------------------

@given(s=series_of(nullable_scalars))
@settings(max_examples=100)
def test_codes_are_valid_indices(s):
    codes, uniques = s.factorize()
    non_sentinel = codes[codes >= 0]
    assert (non_sentinel >= 0).all()
    assert (non_sentinel < len(uniques)).all()


# ---------------------------------------------------------------------------
# 4. Sentinel -1 assigned to NaN when use_na_sentinel=True (default)
# ---------------------------------------------------------------------------

@given(
    values=st.lists(
        st.one_of(st.floats(allow_nan=False, allow_infinity=False, min_value=-100, max_value=100), st.just(np.nan)),
        min_size=2, max_size=12,
    )
)
@settings(max_examples=100)
def test_nan_gets_sentinel_minus_one(values):
    s = pd.Series(values, dtype=float)
    codes, uniques = s.factorize(use_na_sentinel=True)
    nan_mask = s.isna().values
    assert (codes[nan_mask] == -1).all()
    # non-NaN entries should not be -1
    assert (codes[~nan_mask] >= 0).all()


# ---------------------------------------------------------------------------
# 5. No NaN in uniques when use_na_sentinel=True
# ---------------------------------------------------------------------------

@given(
    values=st.lists(
        st.one_of(st.floats(allow_nan=False, allow_infinity=False, min_value=-100, max_value=100), st.just(np.nan)),
        min_size=1, max_size=12,
    )
)
@settings(max_examples=100)
def test_no_nan_in_uniques_with_sentinel(values):
    s = pd.Series(values, dtype=float)
    _, uniques = s.factorize(use_na_sentinel=True)
    assert not uniques.isna().any()


# ---------------------------------------------------------------------------
# 6. NaN gets non-negative code when use_na_sentinel=False; NaN in uniques
# ---------------------------------------------------------------------------

def test_nan_in_uniques_when_no_sentinel():
    s = pd.Series([1.0, np.nan, 2.0, np.nan])
    codes, uniques = s.factorize(use_na_sentinel=False)
    assert (codes >= 0).all()
    assert uniques.isna().any()


@given(
    values=st.lists(
        st.one_of(st.floats(allow_nan=False, allow_infinity=False, min_value=-100, max_value=100), st.just(np.nan)),
        min_size=2, max_size=12,
    )
)
@settings(max_examples=100)
def test_nan_code_nonnegative_no_sentinel(values):
    assume(any(v != v for v in values))  # at least one NaN (NaN != NaN)
    s = pd.Series(values, dtype=float)
    codes, uniques = s.factorize(use_na_sentinel=False)
    assert (codes >= 0).all()


# ---------------------------------------------------------------------------
# 7. len(uniques) equals number of distinct non-NaN values (use_na_sentinel=True)
# ---------------------------------------------------------------------------

@given(s=series_of(nullable_scalars))
@settings(max_examples=100)
def test_len_uniques_equals_distinct_non_nan(s):
    codes, uniques = s.factorize(use_na_sentinel=True)
    distinct_non_nan = s.dropna().nunique()
    assert len(uniques) == distinct_non_nan


# ---------------------------------------------------------------------------
# 8. Each distinct value maps to exactly one code (consistency)
# ---------------------------------------------------------------------------

@given(s=series_of(scalar_elements))
@settings(max_examples=100)
def test_same_value_same_code(s):
    codes, uniques = s.factorize()
    for val in s.unique():
        mask = (s == val).values
        assert len(set(codes[mask])) == 1, f"value {val!r} maps to multiple codes"


# ---------------------------------------------------------------------------
# 9. sort=False: uniques appear in order of first occurrence
# ---------------------------------------------------------------------------

@given(s=series_of(scalar_elements))
@settings(max_examples=100)
def test_sort_false_first_occurrence_order(s):
    codes, uniques = s.factorize(sort=False)
    seen = []
    seen_set = set()
    for val in s:
        if val not in seen_set:
            seen.append(val)
            seen_set.add(val)
    expected = pd.Index(seen)
    pd.testing.assert_index_equal(uniques, expected)


# ---------------------------------------------------------------------------
# 10. sort=True: uniques are sorted
# ---------------------------------------------------------------------------

@given(s=st.lists(st.integers(-50, 50), min_size=1, max_size=15).map(pd.Series))
@settings(max_examples=100)
def test_sort_true_uniques_sorted_integers(s):
    _, uniques = s.factorize(sort=True)
    sorted_uniques = uniques.sort_values()
    pd.testing.assert_index_equal(uniques, sorted_uniques)


@given(s=st.lists(st.text(alphabet="abcde", min_size=1, max_size=3), min_size=1, max_size=15).map(pd.Series))
@settings(max_examples=100)
def test_sort_true_uniques_sorted_strings(s):
    _, uniques = s.factorize(sort=True)
    sorted_uniques = uniques.sort_values()
    pd.testing.assert_index_equal(uniques, sorted_uniques)


# ---------------------------------------------------------------------------
# 11. sort=True vs sort=False: same set of uniques, consistent codes
# ---------------------------------------------------------------------------

@given(s=series_of(scalar_elements))
@settings(max_examples=100)
def test_sort_true_and_false_same_unique_set(s):
    _, uniques_sorted = s.factorize(sort=True)
    _, uniques_unsorted = s.factorize(sort=False)
    assert set(uniques_sorted.tolist()) == set(uniques_unsorted.tolist())


@given(s=series_of(scalar_elements))
@settings(max_examples=100)
def test_sort_flag_codes_reconstruct_same_values(s):
    codes_t, uniques_t = s.factorize(sort=True)
    codes_f, uniques_f = s.factorize(sort=False)
    recon_t = pd.Index(uniques_t.take(codes_t))
    recon_f = pd.Index(uniques_f.take(codes_f))
    pd.testing.assert_index_equal(recon_t, recon_f)


# ---------------------------------------------------------------------------
# 12. Empty Series: codes is empty, uniques is empty Index
# ---------------------------------------------------------------------------

def test_empty_series():
    s = pd.Series([], dtype=object)
    codes, uniques = s.factorize()
    assert len(codes) == 0
    assert len(uniques) == 0
    assert isinstance(uniques, pd.Index)


# ---------------------------------------------------------------------------
# 13. All-NaN Series with use_na_sentinel=True: all codes -1, uniques empty
# ---------------------------------------------------------------------------

def test_all_nan_sentinel_true():
    s = pd.Series([np.nan, np.nan, np.nan])
    codes, uniques = s.factorize(use_na_sentinel=True)
    assert (codes == -1).all()
    assert len(uniques) == 0


# ---------------------------------------------------------------------------
# 14. All-NaN Series with use_na_sentinel=False: non-negative codes, NaN in uniques
# ---------------------------------------------------------------------------

def test_all_nan_sentinel_false():
    s = pd.Series([np.nan, np.nan, np.nan])
    codes, uniques = s.factorize(use_na_sentinel=False)
    assert (codes >= 0).all()
    assert uniques.isna().any()


# ---------------------------------------------------------------------------
# 15. Single-element Series: one unique, code is 0
# ---------------------------------------------------------------------------

@given(val=scalar_elements)
@settings(max_examples=50)
def test_single_element_series(val):
    s = pd.Series([val])
    codes, uniques = s.factorize()
    assert len(uniques) == 1
    assert codes[0] == 0


# ---------------------------------------------------------------------------
# 16. Repeated single value: all codes identical (and equal to 0)
# ---------------------------------------------------------------------------

@given(
    val=scalar_elements,
    n=st.integers(1, 20),
)
@settings(max_examples=80)
def test_repeated_single_value_all_codes_identical(val, n):
    s = pd.Series([val] * n)
    codes, uniques = s.factorize()
    assert len(uniques) == 1
    assert (codes == codes[0]).all()
    assert codes[0] == 0


# ---------------------------------------------------------------------------
# 17. codes length equals Series length
# ---------------------------------------------------------------------------

@given(s=series_of(nullable_scalars))
@settings(max_examples=100)
def test_codes_length_equals_series_length(s):
    codes, _ = s.factorize()
    assert len(codes) == len(s)


# ---------------------------------------------------------------------------
# 18. Reconstruction with use_na_sentinel=False (NaN included)
# ---------------------------------------------------------------------------

@given(
    values=st.lists(
        st.one_of(st.floats(allow_nan=False, allow_infinity=False, min_value=-10, max_value=10), st.just(np.nan)),
        min_size=1, max_size=12,
    )
)
@settings(max_examples=100)
def test_reconstruction_no_sentinel(values):
    s = pd.Series(values, dtype=float)
    codes, uniques = s.factorize(use_na_sentinel=False)
    reconstructed = uniques.take(codes)
    # Compare element-wise to handle NaN == NaN
    assert len(reconstructed) == len(s)
    for r, orig in zip(reconstructed, s):
        if pd.isna(orig):
            assert pd.isna(r)
        else:
            assert r == orig
