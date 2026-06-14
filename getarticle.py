
import pandas as pd
pd.set_option('display.max_colwidth', None)
real = pd.read_csv('data/True.csv')
article = real.iloc[8900]['title'] + ' ' + real.iloc[3]['text']
print(article[:1000])
