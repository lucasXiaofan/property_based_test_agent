# pandas.api.typing.DataFrameGroupBy.mean

- Source URL: https://pandas.pydata.org/pandas-docs/version/3.0/reference/api/pandas.api.typing.DataFrameGroupBy.mean.html

# pandas.api.typing.DataFrameGroupBy.mean#

DataFrameGroupBy.mean(numeric_only=False, skipna=True, engine=None, engine_kwargs=None)[source]#
Compute mean of groups, excluding missing values.
Parameters:
numeric_onlybool, default False
Include only float, int, boolean columns.
Changed in version 2.0.0: numeric_only no longer accepts `None` and defaults to `False`.
skipnabool, default True
Exclude NA/null values. If an entire group is NA, the result will be NA.
enginestr, default None
-
`'cython'` : Runs the operation through C-extensions from cython.
-
`'numba'` : Runs the operation through JIT compiled code from numba.
-
`None` : Defaults to `'cython'` or globally setting `compute.use_numba`
engine_kwargsdict, default None
-
For `'cython'` engine, there are no accepted `engine_kwargs`
-
For `'numba'` engine, the engine can accept `nopython`, `nogil` and `parallel` dictionary keys. The values must either be `True` or `False`. The default `engine_kwargs` for the `'numba'` engine is `{'nopython':True,'nogil':False,'parallel':False}`
Returns:
pandas.Series or pandas.DataFrame
Mean of values within each group. Same object type as the caller.
See also
`Series.mean`
Apply function mean to a Series.
`DataFrame.mean`
Apply function mean to each row or column of a DataFrame.
Examples

```text
>>> df = pd.DataFrame(
...     {"A": [1, 1, 2, 1, 2], "B": [np.nan, 2, 3, 4, 5], "C": [1, 2, 1, 1, 2]},
...     columns=["A", "B", "C"],
... )

```
Groupby one column and return the mean of the remaining columns in each group.

```text
>>> df.groupby("A").mean()
     B         C
A
1  3.0  1.333333
2  4.0  1.500000

```
Groupby two columns and return the mean of the remaining column.

```text
>>> df.groupby(["A", "B"]).mean()
         C
A B
1 2.0  2.0
  4.0  1.0
2 3.0  1.0
  5.0  2.0

```
Groupby one column and return the mean of only particular column in the group.

```text
>>> df.groupby("A")["B"].mean()
A
1    3.0
2    4.0
Name: B, dtype: float64

```
