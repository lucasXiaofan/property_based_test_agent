import numpy as np
from pandas import DataFrame

df = DataFrame({"A": range(4, 8), "B": range(4)})

df.ewm(1).aggregate(sum)  # fails
df.ewm(1).aggregate("sum") # works
df.ewm(1).aggregate(np.sum) # fails