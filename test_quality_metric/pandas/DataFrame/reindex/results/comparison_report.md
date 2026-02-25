# Test Suite Comparison: `pandas.DataFrame.reindex`

- Generated at: 2026-02-25T04:05:35Z
- Suite A: `baseline_test`
- Suite B: `ir_generated_test`
- Target dir: `/Users/xiaofanlu/Documents/github_repos/property_based_test_agent/test_quality_metric/pandas/DataFrame/reindex`

---

## 1. Baseline Test Status

### Suite A — `baseline_test`
- Status: **FAIL**
- Failed (4):
  - `test_fill_value_used_for_missing_row_labels`
  - `test_limit_caps_consecutive_fills`
  - `test_missing_row_labels_get_nan`
  - `test_multiindex_level_reindex_adds_nan_for_new_level_value`

### Suite B — `ir_generated_test`
- Status: **PASS**
- All tests passed.

---

## 2. Mutant Kill Comparison

| Mutant ID | Clause | Description | Suite A | Suite B | Change |
|-----------|--------|-------------|---------|---------|--------|
| `M_C10_ignore_limit` | C10 | limit caps the maximum number of consecutive forward/ba | ❌ survived | ✅ killed | 🆕 B kills |
| `M_C11_break_index_columns_convention` | C11 | DataFrame.reindex supports keyword convention using ind | ✅ killed | ✅ killed |  |
| `M_C12_break_labels_axis_convention` | C12 | DataFrame.reindex supports labels with explicit axis co | ✅ killed | ✅ killed |  |
| `M_C13_axis_int_rejected` | C13 | axis accepts both axis names and numeric selectors. | ✅ killed | ❌ survived | 🔻 B loses |
| `M_C14_columns_missing_not_nan` | C14 | columns reindexing may introduce NaN for missing column | ✅ killed | ✅ killed |  |
| `M_C15_scalar_tolerance_not_uniform` | C15 | Scalar tolerance applies uniformly to all values. | ❌ survived | ❌ survived |  |
| `M_C16_list_tolerance_len_dtype_ignored` | C16 | List-like tolerance must match index size and exact ind | ❌ survived | ✅ killed | 🆕 B kills |
| `M_C17_tolerance_inequality_relaxed` | C17 | Tolerance matching follows the absolute-distance inequa | ❌ survived | ✅ killed | 🆕 B kills |
| `M_C18_ignore_level_param` | C18 | level performs matching across the specified MultiIndex | ✅ killed | ❌ survived | 🔻 B loses |
| `M_C19_copy_changes_behavior` | C19 | copy keyword is ignored and has no effect on method beh | ❌ survived | ✅ killed | 🆕 B kills |
| `M_C1_not_conformed_return` | C1 | reindex returns a DataFrame conformed to the requested  | ✅ killed | ✅ killed |  |
| `M_C20_force_materialize_non_index` | C20 | Using Index objects for labels/index/columns is preferr | ❌ survived | ❌ survived |  |
| `M_C2_missing_labels_not_nan` | C2 | Labels introduced by reindex are NaN by default when va | ✅ killed | ✅ killed |  |
| `M_C3_ignore_fill_value` | C3 | fill_value replaces missing values introduced by reinde | ✅ killed | ✅ killed |  |
| `M_C4_fill_value_numeric_only` | C4 | fill_value supports non-numeric compatible values. | ❌ survived | ❌ survived |  |
| `M_C5_ffill_backward` | C5 | method='ffill'/'pad' propagates last valid observation  | ✅ killed | ✅ killed |  |
| `M_C6_bfill_forward` | C6 | method='bfill'/'backfill' propagates next valid observa | ✅ killed | ✅ killed |  |
| `M_C7_nearest_as_ffill` | C7 | method='nearest' fills gaps using nearest valid observa | ❌ survived | ✅ killed | 🆕 B kills |
| `M_C8_method_without_monotonicity` | C8 | method-based filling applies only to monotonic indexes. | ✅ killed | ✅ killed |  |
| `M_C9_fill_existing_nan` | C9 | method-based filling does not fill pre-existing NaN val | ✅ killed | ✅ killed |  |

### Mutation Score Summary

| Metric | Suite A | Suite B | Delta |
|--------|---------|---------|-------|
| Killed | 12/20 | 15/20 | +3 |
| Survived | 8/20 | 5/20 | -3 |
| **Mutation score** | **0.600** | **0.750** | **+0.150** |

---

## 3. Coverage Comparison

### Method Line Coverage

| Method | Suite A lines | Suite B lines | Suite A % | Suite B % | Delta |
|--------|--------------|--------------|-----------|-----------|-------|
| `reindex` | 1/1 | 1/1 | 100.0% | 100.0% | +0.0pp |
| `reindex` | 16/22 | 14/22 | 72.7% | 63.6% | -9.1pp |
| **Average** | | | **86.37%** | **81.82%** | **-4.55pp** |

### Branch Coverage

| Metric | Suite A | Suite B | Delta |
|--------|---------|---------|-------|
| Covered branches | 103/1960 | 67/1960 | -36 |
| **Branch coverage** | **5.26%** | **3.42%** | **-1.84pp** |

---

## 4. Properties Coverage Comparison

| Clause | Category | Description | Suite A | Suite B | Change |
|--------|----------|-------------|---------|---------|--------|
| C1 | Return Contract | reindex returns a DataFrame conformed to the requested label | ✅ | ✅ |  |
| C2 | NaN / Missing Label Behavior | Labels introduced by reindex are NaN by default when values  | ✅ | ✅ |  |
| C3 | NaN / Missing Label Behavior | fill_value replaces missing values introduced by reindex. | ✅ | ✅ |  |
| C4 | NaN / Missing Label Behavior | fill_value supports non-numeric compatible values. | ❌ | ❌ |  |
| C5 | Method / Fill Logic | method='ffill'/'pad' propagates last valid observation forwa | ✅ | ✅ |  |
| C6 | Method / Fill Logic | method='bfill'/'backfill' propagates next valid observation  | ✅ | ✅ |  |
| C7 | Method / Fill Logic | method='nearest' fills gaps using nearest valid observation. | ✅ | ✅ |  |
| C8 | Method / Fill Logic | method-based filling applies only to monotonic indexes. | ✅ | ✅ |  |
| C9 | Method / Fill Logic | method-based filling does not fill pre-existing NaN values i | ✅ | ✅ |  |
| C10 | Method / Fill Logic | limit caps the maximum number of consecutive forward/backwar | ✅ | ✅ |  |
| C11 | Axis / Label Targeting | DataFrame.reindex supports keyword convention using index= a | ✅ | ✅ |  |
| C12 | Axis / Label Targeting | DataFrame.reindex supports labels with explicit axis convent | ✅ | ❌ | 🔻 B loses |
| C13 | Axis / Label Targeting | axis accepts both axis names and numeric selectors. | ❌ | ❌ |  |
| C14 | Axis / Label Targeting | columns reindexing may introduce NaN for missing columns. | ✅ | ✅ |  |
| C15 | Tolerance | Scalar tolerance applies uniformly to all values. | ❌ | ✅ | 🆕 B covers |
| C16 | Tolerance | List-like tolerance must match index size and exact index dt | ❌ | ❌ |  |
| C17 | Tolerance | Tolerance matching follows the absolute-distance inequality. | ❌ | ✅ | 🆕 B covers |
| C18 | Level | level performs matching across the specified MultiIndex leve | ✅ | ❌ | 🔻 B loses |
| C19 | Deprecated | copy keyword is ignored and has no effect on method behavior | ❌ | ✅ | 🆕 B covers |
| C20 | Performance / Input Shape | Using Index objects for labels/index/columns is preferred to | ❌ | ❌ |  |

### Properties Coverage Summary

| Metric | Suite A | Suite B | Delta |
|--------|---------|---------|-------|
| Satisfied | 13/20 | 14/20 | +1 |
| **Coverage** | **65.0%** | **70.0%** | **+5.0pp** |

---

## 5. Summary

| Metric | Suite A | Suite B | Delta | Better |
|--------|---------|---------|-------|--------|
| Baseline passes | ❌ FAIL (4 tests) | ✅ PASS | — | — |
| Mutation score | 0.600 | 0.750 | +0.150 | **B** |
| Properties coverage | 65.0% | 70.0% | +5.0pp | **B** |
| Avg method line cov | 86.4% | 81.8% | -4.5pp | **A** |
| Branch coverage | 5.3% | 3.4% | -1.8pp | **A** |
