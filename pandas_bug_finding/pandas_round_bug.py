# reference https://pandas.pydata.org/pandas-docs/version/2.2/reference/api/pandas.Series.round.html

import pandas as pd 
import pytest
from hypothesis import given, strategies as st

@given(data = st.lists(st.text(),min_size = 1))
def test_round_failed_on_text_columns(data):
    s = pd.Series(data,dtype = "object")
    with pytest.raises(TypeError):
        s.round()

if __name__ == "__main__":
    import numpy as np 
    data = {'bar': ['foo', 0.2]}
    df = pd.DataFrame(data)
    print(df['bar'])
    print(df['bar'].round())
    
    multi = pd.DataFrame(data = [0.9,0.03,0.1231],columns = ['bar'])
    print(multi['bar'])
    print(multi['bar'].round())

    df=pd.DataFrame(data=['foo'],columns=['bar'])
    print(df['bar'])

    
    # df.loc[0,'bar']=0.2
    # print(df['bar'])
    # print(df['bar'].round())
    # print(np.round(df['bar']))
    # import numpy as np 
    # data = ["1", "2", "3"]
    # np_result = np.round(np.array(data, dtype=object))
    # print(np_result)
