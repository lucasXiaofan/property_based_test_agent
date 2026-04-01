# pandas.Series.str.match

- Source URL: https://pandas.pydata.org/pandas-docs/version/3.0/reference/api/pandas.Series.str.match.html

# pandas.Series.str.match#

Series.str.match(pat, case=<no_default>, flags=<no_default>, na=<no_default>)[source]#
Determine if each string starts with a match of a regular expression.
Determines whether each string in the Series or Index starts with a match to a specified regular expression. This function is especially useful for validating prefixes, such as ensuring that codes, tags, or identifiers begin with a specific pattern.
Parameters:
patstr or compiled regex
Character sequence or regular expression.
casebool, default True
If True, case sensitive.
flagsint, default 0 (no flags)
Regex module flags, e.g. re.IGNORECASE.
nascalar, optional
Fill value for missing values. The default depends on dtype of the array. For the `"str"` dtype, `False` is used. For object dtype, `numpy.nan` is used. For the nullable `StringDtype`, `pandas.NA` is used.
Returns:
Series/Index/array of boolean values
A Series, Index, or array of boolean values indicating whether the start of each string matches the pattern. The result will be of the same type as the input.
See also
`fullmatch`
Stricter matching that requires the entire string to match.
`contains`
Analogous, but less strict, relying on re.search instead of re.match.
`extract`
Extract matched groups.
Examples

```text
>>> ser = pd.Series(["horse", "eagle", "donkey"])
>>> ser.str.match("e")
0   False
1   True
2   False
dtype: bool

```
