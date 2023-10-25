import pandas as pd

df = pd.read_csv('./data/test.csv', 
        names=['year', 'month', 'day', 'week', 'day_2',
               'day_1', 'average', 'actual'], header=0)
print(df.iloc[0])
