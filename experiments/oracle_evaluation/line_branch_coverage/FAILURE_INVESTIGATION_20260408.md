# Failure Investigation 2026-04-08

Initial batch coverage run exposed four failing suites. The failures were in test expectations, not in the local coverage runner.

## Fixed failures

### `Index/astype` `ir_generated_test.py`

- Problem: the test expected only `TypeError` for invalid `datetime64[ns]` casts from arbitrary strings.
- Actual behavior: pandas 3.0.0 can raise `TypeError` or `ValueError`-family parse errors depending on the string.
- Fix: accept both `TypeError` and `ValueError`.

### `Series/mean` `baseline_test.py`

- Problem: the bounded-by-`[min, max]` assertion used a fixed `1e-9` tolerance.
- Actual behavior: large floating-point values can differ by more than that due to rounding, even when the mean is numerically valid.
- Fix: use a scale-aware tolerance.

### `Series.str/contains` `ir_generated_test.py`

- Problem: the test assumed omitted `na` would preserve missing values for a default-inferred Series.
- Actual behavior: the default missing-value result depends on dtype.
- Fix: force `dtype=object` in the test so the asserted NaN-propagation behavior matches pandas semantics.

### `Series.str/match` `ir_generated_test.py`

- Problem: several tests created all-`NaN` or mixed string/`NaN` Series without forcing object dtype, so pandas inferred floating dtype in some cases and rejected `.str`.
- Actual behavior: `.str` accessor is invalid on floating-only Series.
- Fix: force `dtype=object` for these null-handling tests.
