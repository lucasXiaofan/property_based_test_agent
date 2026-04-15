# pandas.Series.mean

- Source URL: https://pandas.pydata.org/pandas-docs/version/3.0/reference/api/pandas.Series.mean.html

# pandas.Series.mean#

Series.mean(*, axis=0, skipna=True, numeric_only=False, **kwargs)[source]#
Return the mean of the values over the requested axis.
Parameters:
axis{index (0)}
Axis for the function to be applied on. For Series this parameter is unused and defaults to 0.
For DataFrames, specifying `axis=None` will apply the aggregation across both axes.
Added in version 2.0.0.
skipnabool, default True
Exclude NA/null values when computing the result.
numeric_onlybool, default False
Include only float, int, boolean columns.
**kwargs
Additional keyword arguments to be passed to the function.
Returns:
scalar or Series (if level specified)
Mean of the values for the requested axis.
See also
`numpy.median`
Equivalent numpy function for computing median.
`Series.sum`
Sum of the values.
`Series.median`
Median of the values.
`Series.std`
Standard deviation of the values.
`Series.var`
Variance of the values.
`Series.min`
Minimum value.
`Series.max`
Maximum value.
Examples

```text
>>> s = pd.Series([1, 2, 3])
>>> s.mean()
2.0

```
