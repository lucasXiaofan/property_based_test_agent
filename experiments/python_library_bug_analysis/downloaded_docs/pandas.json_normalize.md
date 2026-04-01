# pandas.json_normalize

- Source URL: https://pandas.pydata.org/pandas-docs/version/3.0/reference/api/pandas.json_normalize.html

# pandas.json_normalize#

pandas.json_normalize(data, record_path=None, meta=None, meta_prefix=None, record_prefix=None, errors='raise', sep='.', max_level=None)[source]#
Normalize semi-structured JSON data into a flat table.
This method is designed to transform semi-structured JSON data, such as nested dictionaries or lists, into a flat table. This is particularly useful when handling JSON-like data structures that contain deeply nested fields.
Parameters:
datadict, list of dicts, or Series of dicts
Unserialized JSON objects.
record_pathstr or list of str, default None
Path in each object to list of records. If not passed, data will be assumed to be an array of records.
metalist of paths (str or list of str), default None
Fields to use as metadata for each record in resulting table.
meta_prefixstr, default None
String to prefix records with dotted path, e.g. foo.bar.field if meta is [‘foo’, ‘bar’].
record_prefixstr, default None
String to prefix records with dotted path, e.g. foo.bar.field if path to records is [‘foo’, ‘bar’].
errors{‘raise’, ‘ignore’}, default ‘raise’
Configures error handling.
-
‘ignore’ : will ignore KeyError if keys listed in meta are not always present.
-
‘raise’ : will raise KeyError if keys listed in meta are not always present.
sepstr, default ‘.’
Nested records will generate names separated by sep. e.g., for sep=’.’, {‘foo’: {‘bar’: 0}} -> foo.bar.
max_levelint, default None
Max number of levels(depth of dict) to normalize. if None, normalizes all levels.
Returns:
DataFrame
The normalized data, represented as a pandas DataFrame.
See also
`DataFrame`
Two-dimensional, size-mutable, potentially heterogeneous tabular data.
`Series`
One-dimensional ndarray with axis labels (including time series).
Examples

```text
>>> data = [
...     {"id": 1, "name": {"first": "Coleen", "last": "Volk"}},
...     {"name": {"given": "Mark", "family": "Regner"}},
...     {"id": 2, "name": "Faye Raker"},
... ]
>>> pd.json_normalize(data)
    id name.first name.last name.given name.family        name
0  1.0     Coleen      Volk        NaN         NaN         NaN
1  NaN        NaN       NaN       Mark      Regner         NaN
2  2.0        NaN       NaN        NaN         NaN  Faye Raker

```

```text
>>> data = [
...     {
...         "id": 1,
...         "name": "Cole Volk",
...         "fitness": {"height": 130, "weight": 60},
...     },
...     {"name": "Mark Reg", "fitness": {"height": 130, "weight": 60}},
...     {
...         "id": 2,
...         "name": "Faye Raker",
...         "fitness": {"height": 130, "weight": 60},
...     },
... ]
>>> pd.json_normalize(data, max_level=0)
    id        name                        fitness
0  1.0   Cole Volk  {'height': 130, 'weight': 60}
1  NaN    Mark Reg  {'height': 130, 'weight': 60}
2  2.0  Faye Raker  {'height': 130, 'weight': 60}

```
Normalizes nested data up to level 1.

```text
>>> data = [
...     {
...         "id": 1,
...         "name": "Cole Volk",
...         "fitness": {"height": 130, "weight": 60},
...     },
...     {"name": "Mark Reg", "fitness": {"height": 130, "weight": 60}},
...     {
...         "id": 2,
...         "name": "Faye Raker",
...         "fitness": {"height": 130, "weight": 60},
...     },
... ]
>>> pd.json_normalize(data, max_level=1)
    id        name  fitness.height  fitness.weight
0  1.0   Cole Volk             130              60
1  NaN    Mark Reg             130              60
2  2.0  Faye Raker             130              60

```

```text
>>> data = [
...     {
...         "id": 1,
...         "name": "Cole Volk",
...         "fitness": {"height": 130, "weight": 60},
...     },
...     {"name": "Mark Reg", "fitness": {"height": 130, "weight": 60}},
...     {
...         "id": 2,
...         "name": "Faye Raker",
...         "fitness": {"height": 130, "weight": 60},
...     },
... ]
>>> series = pd.Series(data, index=pd.Index(["a", "b", "c"]))
>>> pd.json_normalize(series)
    id        name  fitness.height  fitness.weight
a  1.0   Cole Volk             130              60
b  NaN    Mark Reg             130              60
c  2.0  Faye Raker             130              60

```

```text
>>> data = [
...     {
...         "state": "Florida",
...         "shortname": "FL",
...         "info": {"governor": "Rick Scott"},
...         "counties": [
...             {"name": "Dade", "population": 12345},
...             {"name": "Broward", "population": 40000},
...             {"name": "Palm Beach", "population": 60000},
...         ],
...     },
...     {
...         "state": "Ohio",
...         "shortname": "OH",
...         "info": {"governor": "John Kasich"},
...         "counties": [
...             {"name": "Summit", "population": 1234},
...             {"name": "Cuyahoga", "population": 1337},
...         ],
...     },
... ]
>>> result = pd.json_normalize(
...     data, "counties", ["state", "shortname", ["info", "governor"]]
... )
>>> result
         name  population    state shortname info.governor
0        Dade       12345   Florida    FL    Rick Scott
1     Broward       40000   Florida    FL    Rick Scott
2  Palm Beach       60000   Florida    FL    Rick Scott
3      Summit        1234   Ohio       OH    John Kasich
4    Cuyahoga        1337   Ohio       OH    John Kasich

```

```text
>>> data = {"A": [1, 2]}
>>> pd.json_normalize(data, "A", record_prefix="Prefix.")
    Prefix.0
0          1
1          2

```
Returns normalized data with columns prefixed with the given string.
