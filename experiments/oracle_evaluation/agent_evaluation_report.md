# Agent Evaluation Report

- Generated at: `2026-04-01T10:06:20.599882-04:00`
- Pandas version: `3.0.0`
- JSON details: `/Users/xiaofanlu/Documents/github_repos/property_based_test_agent/experiments/oracle_evaluation/bug_trigger_evaluation_20260401T100620-0400.json`
- Scoring: `yes=1.0`, `no=0.0`

## Overall

- Baseline trigger coverage: `0.000` (0.0 over 10 issue-links; yes=0)
- IR-generated trigger coverage: `0.100` (1.0 over 10 issue-links; yes=1)
- Higher-quality suite by bug-trigger coverage: `ir_generated`

This report does not depend on whether the local pandas wheel still reproduces the bug. It checks whether each suite actually exercises the reported buggy input or condition from the counted issue inventory.

## Per API

### pandas.DataFrame.groupby

- Baseline: `0.000` (score=0.0, yes=0)
- IR-generated: `0.000` (score=0.0, yes=0)
- Higher quality: `tie`

- Issue `#61356`: groupby.groups fails with categorical + NaN + dropna=False
  Trigger: Categorical grouper containing NaN, grouped with `dropna=False`, then inspect `.groups` / NA-bucket handling.
  Baseline: `no`. The baseline suite does not directly combine categorical keys, NaN, `dropna=False`, and `.groups` inspection in one test, so it does not match the reported trigger.
  IR-generated: `no`. The IR suite covers related pieces, but not the exact `.groups` failure path with categorical NaNs under `dropna=False`, so it is not a direct trigger match.

### pandas.DataFrame.reindex

- Baseline: `0.000` (score=0.0, yes=0)
- IR-generated: `0.000` (score=0.0, yes=0)
- Higher quality: `tie`

- Issue `#63993`: DataFrame.reindex crashes with multi-column string fill_value
  Trigger: Reindex columns so that at least two new columns are introduced while using a string `fill_value`, which previously crashed.
  Baseline: `no`. The baseline reindex suite uses numeric fill values and tests column insertion separately, so it never combines multi-column column reindexing with a string `fill_value`.
  IR-generated: `no`. The IR suite uses `fill_value='missing'` only for row reindexing and does not combine it with multi-column column reindexing, so it does not directly match the reported trigger.

### pandas.DataFrame.to_json

- Baseline: `0.000` (score=0.0, yes=0)
- IR-generated: `0.000` (score=0.0, yes=0)
- Higher quality: `tie`

- Issue `#63236`: to_json stringifies non-ns TimedeltaIndex with wrong units
  Trigger: Serialize a DataFrame whose column labels are non-nanosecond `TimedeltaIndex` values and verify unit-preserving JSON output.
  Baseline: `no`. The baseline to_json suite checks orient structure, precision, JSON nulls, and datetime formatting, but not TimedeltaIndex column-label serialization.
  IR-generated: `no`. The IR to_json suite checks epoch scaling for datetime values, not non-nanosecond TimedeltaIndex column labels.

### pandas.Index.astype

- Baseline: `0.000` (score=0.0, yes=0)
- IR-generated: `0.000` (score=0.0, yes=0)
- Higher quality: `tie`

- Issue `#61099`: Series comparison fails for object-index vs string-index
  Trigger: Build a Series with an object index, convert a sibling Index to nullable string dtype with `astype('string')`, then compare the Series objects.
  Baseline: `no`. The baseline astype suite validates dtype conversion, copy behavior, and impossible casts, but it never converts to nullable string dtype and never performs downstream Series comparison.
  IR-generated: `no`. The IR astype suite focuses on numeric/object conversions and copy semantics only; it never creates a string-dtype Index and compares Series indexed by object vs string indexes.

### pandas.Index.shift

- Baseline: `0.000` (score=0.0, yes=0)
- IR-generated: `0.000` (score=0.0, yes=0)
- Higher quality: `tie`

- Issue `#62094`: TimedeltaIndex.shift regressed on computed freq-less indexes
  Trigger: Create a TimedeltaIndex by subtracting a Timestamp from a date range, producing a computed freq-less index, then call `shift(1)`.
  Baseline: `no`. The baseline shift suite uses `date_range` and `timedelta_range` constructors with explicit frequencies; it never constructs the computed freq-less TimedeltaIndex from timestamp subtraction.
  IR-generated: `no`. The IR suite tests freqless TimedeltaIndex behavior, but not the reported computed index produced by datetime arithmetic, so it does not directly match the trigger.

### pandas.Series.factorize

- Baseline: `0.000` (score=0.0, yes=0)
- IR-generated: `0.000` (score=0.0, yes=0)
- Higher quality: `tie`

- Issue `#62888`: factorize collapses 0/False and 1/True in object dtype
  Trigger: Object-dtype Series mixing `0`, `1`, `False`, and `True`, then check whether factorization preserves four distinct values.
  Baseline: `no`. The baseline factorize suite never mixes ints and bools in the same object Series, so it does not hit the hash/equality collision that drives the bug.
  IR-generated: `no`. The IR factorize suite focuses on strings, missing values, and categoricals, not mixed int/bool object inputs.

### pandas.Series.mean

- Baseline: `0.000` (score=0.0, yes=0)
- IR-generated: `0.000` (score=0.0, yes=0)
- Higher quality: `tie`

- Issue `#59965`: FloatingArray reductions do not skip NaN correctly
  Trigger: Nullable FloatingArray / convert_dtypes input mixed with missing values, then reduction semantics with skipna handling.
  Baseline: `no`. The baseline suite checks `skipna=True` and `skipna=False` with NaNs, but it does not directly use nullable FloatingArray / `convert_dtypes()` inputs, so it does not match the reported trigger.
  IR-generated: `no`. The IR suite stresses skipna behavior, but it still uses plain float/bool Series rather than nullable FloatingArray inputs, so it does not match the reported trigger.

### pandas.Series.mul

- Baseline: `0.000` (score=0.0, yes=0)
- IR-generated: `0.000` (score=0.0, yes=0)
- Higher quality: `tie`

- Issue `#62595`: Arrow-backed string Series multiply behaves differently from python strings
  Trigger: Multiply a string Series by boolean values and compare behavior across string backends, especially arrow-backed strings.
  Baseline: `no`. The baseline mul suite is entirely numeric and never exercises string Series, bool operands, or backend-specific string semantics.
  IR-generated: `no`. The IR mul suite is also numeric-only, so it misses the string-backend and bool-multiplication trigger entirely.

### pandas.Series.str.contains

- Baseline: `0.000` (score=0.0, yes=0)
- IR-generated: `0.000` (score=0.0, yes=0)
- Higher quality: `tie`

- Issue `#62240`: Compiled regex handling in str.match/str.contains is inconsistent
  Trigger: Pass a compiled regex object, especially one carrying `re.IGNORECASE`, through `Series.str.contains` and compare to expected regex semantics.
  Baseline: `no`. The baseline contains suite never passes a compiled `re.Pattern` object as `pat`, so it does not directly match the reported trigger.
  IR-generated: `no`. The IR contains suite still omits compiled regex objects, so it does not directly match the reported trigger.

### pandas.Series.str.match

- Baseline: `0.000` (score=0.0, yes=0)
- IR-generated: `1.000` (score=1.0, yes=1)
- Higher quality: `ir_generated`

- Issue `#62240`: Compiled regex handling in str.match/str.contains is inconsistent
  Trigger: Pass a compiled regex object, including one with embedded `re.IGNORECASE`, through `Series.str.match` and verify behavior matches Python regex semantics.
  Baseline: `no`. The baseline match suite compares compiled regex vs string patterns, but not with embedded `re.IGNORECASE`, so it does not directly match the reported trigger.
  IR-generated: `yes`. The IR match suite directly tests both compiled regex parity and a compiled `re.IGNORECASE` pattern, which captures the reported buggy condition.
