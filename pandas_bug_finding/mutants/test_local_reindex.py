import sys
sys.path.insert(0, "/Users/xiaofanlu/Documents/github_repos/property_based_test_agent/pandas_bug_finding/pandas")

import pandas as pd
import numpy as np

df = pd.DataFrame({"A": [10, 20, 30]}, index=[0, 1, 2])

print(df.reindex([1, 2, 3, 4]))
print(df.reindex([1, 2, 3, 4], fill_value=0))
