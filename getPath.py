import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')
print(df.head())

import os
print(os.getcwd())
print(os.listdir(os.getcwd()))