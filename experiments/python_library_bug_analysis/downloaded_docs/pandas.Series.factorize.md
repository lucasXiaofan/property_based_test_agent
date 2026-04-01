# pandas.Series.factorize

- Source URL: https://pandas.pydata.org/pandas-docs/version/3.0/reference/api/pandas.Series.factorize.html

# pandas.Series.factorize#

Series.factorize(sort=False, use_na_sentinel=True)[source]#
Encode the object as an enumerated type or categorical variable.
This method is useful for obtaining a numeric representation of an array when all that matters is identifying distinct values. factorize is available as both a top-level function `pandas.factorize()`, and as a method `Series.factorize()` and `Index.factorize()`.
Parameters:
sortbool, default False
Sort uniques and shuffle codes to maintain the relationship.
use_na_sentinelbool, default True
If True, the sentinel -1 will be used for NaN values. If False, NaN values will be encoded as non-negative integers and will not drop the NaN from the uniques of the values.
Returns:
codesndarray
An integer ndarray that’s an indexer into uniques. `uniques.take(codes)` will have the same values as values.
uniquesndarray, Index, or Categorical
The unique valid values. When values is Categorical, uniques is a Categorical. When values is some other pandas object, an Index is returned. Otherwise, a 1-D ndarray is returned.
Note
Even if there’s a missing value in values, uniques will not contain an entry for it.
See also
`cut`
Discretize continuous-valued array.
`unique`
Find the unique value in an array.
Notes
Reference the user guide for more examples.
Examples
These examples all show factorize as a top-level method like `pd.factorize(values)`. The results are identical for methods like `Series.factorize()`.

```text
>>> codes, uniques = pd.factorize(
...     np.array(["b", "b", "a", "c", "b"], dtype="O")
... )
>>> codes
array([0, 0, 1, 2, 0])
>>> uniques
array(['b', 'a', 'c'], dtype=object)

```
With `sort=True`, the uniques will be sorted, and codes will be shuffled so that the relationship is the maintained.

```text
>>> codes, uniques = pd.factorize(
...     np.array(["b", "b", "a", "c", "b"], dtype="O"), sort=True
... )
>>> codes
array([1, 1, 0, 2, 1])
>>> uniques
array(['a', 'b', 'c'], dtype=object)

```
When `use_na_sentinel=True` (the default), missing values are indicated in the codes with the sentinel value `-1` and missing values are not included in uniques.

```text
>>> codes, uniques = pd.factorize(
...     np.array(["b", None, "a", "c", "b"], dtype="O")
... )
>>> codes
array([ 0, -1,  1,  2,  0])
>>> uniques
array(['b', 'a', 'c'], dtype=object)

```
Thus far, we’ve only factorized lists (which are internally coerced to NumPy arrays). When factorizing pandas objects, the type of uniques will differ. For Categoricals, a Categorical is returned.

```text
>>> cat = pd.Categorical(["a", "a", "c"], categories=["a", "b", "c"])
>>> codes, uniques = pd.factorize(cat)
>>> codes
array([0, 0, 1])
>>> uniques
['a', 'c']
Categories (3, str): ['a', 'b', 'c']

```
Notice that `'b'` is in `uniques.categories`, despite not being present in `cat.values`.
For all other pandas objects, an Index of the appropriate type is returned.

```text
>>> cat = pd.Series(["a", "a", "c"])
>>> codes, uniques = pd.factorize(cat)
>>> codes
array([0, 0, 1])
>>> uniques
Index(['a', 'c'], dtype='str')

```
If NaN is in the values, and we want to include NaN in the uniques of the values, it can be achieved by setting `use_na_sentinel=False`.

```text
>>> values = np.array([1, 2, 1, np.nan])
>>> codes, uniques = pd.factorize(values)  # default: use_na_sentinel=True
>>> codes
array([ 0,  1,  0, -1])
>>> uniques
array([1., 2.])

```

```text
>>> codes, uniques = pd.factorize(values, use_na_sentinel=False)
>>> codes
array([0, 1, 0, 2])
>>> uniques
array([ 1.,  2., nan])

```
