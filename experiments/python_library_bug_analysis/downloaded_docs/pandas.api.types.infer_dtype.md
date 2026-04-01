# pandas.api.types.infer_dtype

- Source URL: https://pandas.pydata.org/pandas-docs/version/3.0/reference/api/pandas.api.types.infer_dtype.html

# pandas.api.types.infer_dtype#

pandas.api.types.infer_dtype(value, skipna=True)#
Return a string label of the type of the elements in a list-like input.
This method inspects the elements of the provided input and determines classification of its data type. It is particularly useful for handling heterogeneous data inputs where explicit dtype conversion may not be possible or necessary.
Parameters:
valuelist, ndarray, or pandas type
The input data to infer the dtype.
skipnabool, default True
Ignore NaN values when inferring the type.
Returns:
str
Describing the common type of the input data.
Results can include:
-
string
-
bytes
-
floating
-
integer
-
mixed-integer
-
mixed-integer-float
-
decimal
-
complex
-
categorical
-
boolean
-
datetime64
-
datetime
-
date
-
timedelta64
-
timedelta
-
time
-
period
-
mixed
-
unknown-array
Raises:
TypeError
If ndarray-like but cannot infer the dtype
See also
`api.types.is_scalar`
Check if the input is a scalar.
`api.types.is_list_like`
Check if the input is list-like.
`api.types.is_integer`
Check if the input is an integer.
`api.types.is_float`
Check if the input is a float.
`api.types.is_bool`
Check if the input is a boolean.
Notes
-
‘mixed’ is the catchall for anything that is not otherwise specialized
-
‘mixed-integer-float’ are floats and integers
-
‘mixed-integer’ are integers mixed with non-integers
-
‘unknown-array’ is the catchall for something that is an array (has a dtype attribute), but has a dtype unknown to pandas (e.g. external extension array)
Examples

```text
>>> from pandas.api.types import infer_dtype
>>> infer_dtype(['foo', 'bar'])
'string'

```

```text
>>> infer_dtype(['a', np.nan, 'b'], skipna=True)
'string'

```

```text
>>> infer_dtype(['a', np.nan, 'b'], skipna=False)
'mixed'

```

```text
>>> infer_dtype([b'foo', b'bar'])
'bytes'

```

```text
>>> infer_dtype([1, 2, 3])
'integer'

```

```text
>>> infer_dtype([1, 2, 3.5])
'mixed-integer-float'

```

```text
>>> infer_dtype([1.0, 2.0, 3.5])
'floating'

```

```text
>>> infer_dtype(['a', 1])
'mixed-integer'

```

```text
>>> from decimal import Decimal
>>> infer_dtype([Decimal(1), Decimal(2.0)])
'decimal'

```

```text
>>> infer_dtype([True, False])
'boolean'

```

```text
>>> infer_dtype([True, False, np.nan])
'boolean'

```

```text
>>> infer_dtype([pd.Timestamp('20130101')])
'datetime'

```

```text
>>> import datetime
>>> infer_dtype([datetime.date(2013, 1, 1)])
'date'

```

```text
>>> infer_dtype([np.datetime64('2013-01-01')])
'datetime64'

```

```text
>>> infer_dtype([datetime.timedelta(0, 1, 1)])
'timedelta'

```

```text
>>> infer_dtype(pd.Series(list('aabc')).astype('category'))
'categorical'

```
