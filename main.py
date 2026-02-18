import pandas as pd
import numpy as np

# df = pd.DataFrame(data={"a": [0],"b": [0]})
# print(df)
# print("...")
# print(df.reindex(columns=["a", "b", "c",'d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'], fill_value="missing"))

# df = pd.DataFrame(data={"a": [0],"b": [0]})
# print(df)
# print("...")
# print(df.reindex(columns=["a", "c"], fill_value="missing"))

### I cannot reproduce this bug on mac
# import pandas as pd
# examples=['ga',
#           'Áa',
#           '永a',
#           '🐍a']
# for t in examples:
#     objects = pd.Series([t], dtype=object)
#     strs = pd.Series([t], dtype=str)
#     print('python:', t.find('a'), 
#           'pandas with object type:', objects.str.find('a')[0],
#           'pandas with string type:', strs.str.find('a')[0],          
#           )

# import pandas as pd

# # Pandas object that contains only 1 unique value:
# s = pd.Series(["foo", "foo"])
# s.unique()
# # pandas 2.3.3:
# # array(['foo'], dtype=object)
# #
# # pandas 3.0.0:
# # <ArrowStringArray>
# # ['foo']
# # Length: 1, dtype: str

# # Retrieve the unique value as a scalar
# s.astype(object).unique().item()
# # pandas 2.3.3 or 3.0.0: 'foo'

# # Retrieve the unique value as a scalar
# s.unique().item()
# # pandas 2.3.3: 'foo'
# # pandas 3.0.0: AttributeError: 'ArrowStringArray' object has no attribute 'item'


df = pd.DataFrame({"A": [1.0, 2.0, 3.0]})



print(df.ewm(alpha=0.5).aggregate(np.mean))