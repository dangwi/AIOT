import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

df_train = pd.read_csv('./data/train.csv', 
        names=['year', 'month', 'day', 'week', 'day_2',
               'day_1', 'average', 'actual'], header=0)

df_test  = pd.read_csv('./data/test.csv', 
        names=['year', 'month', 'day', 'week', 'day_2',
               'day_1', 'average', 'actual'], header=0)

# convert df to numpy
data = np.concatenate((df_train.values, df_test.values), axis=0)
print(data.shape)

# split data
y = data[:, -1]

# plot
plt.plot(y, label='actual')
plt.legend()
plt.show()
