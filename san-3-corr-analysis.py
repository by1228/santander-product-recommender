import numpy as np
import pandas as pd
import math
import gc
import seaborn as sns
import matplotlib.pyplot as plt

# Import prepared data
df = pd.read_csv('df_after_lag_feateng.csv')
temp = pd.read_csv('san-col3merge_nonbalanced.csv')

y_colheaders = temp.columns[-24:]
y = df.loc[:,y_colheaders]
X = df.drop(y_colheaders, axis=1)

temp = []

corr = df.corr()

gc.collect()

corr_target_var = corr.loc[y.columns,:].drop(y.columns, axis=1)
sns.set(font_scale=1.2)
fig, ax = plt.subplots()
fig.set_size_inches(25,6)
sns.heatmap(corr_target_var, ax=ax)
plt.ylabel('Products')
plt.xlabel('Customer Features')
fig.savefig('corrtarget.jpg', bbox_inches='tight', dpi=300)

print(abs(corr_target_var).stack().nlargest(50))