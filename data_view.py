# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

#%%
%pylab inline
import seaborn as sbn
import pandas as pd

#%%
data = pd.read_csv("UCI_Credit_Card.csv.zip")
data.head()

#%%
sbn.violinplot(x="default.payment.next.month",
               y="AGE", hue="SEX",
               data=data, split=True,
               palette="cool")

#%%
sbn.violinplot(x="MARRIAGE",
               y="LIMIT_BAL", hue="default.payment.next.month",
               data=data, split=True,
               palette="Set1")

#%%
sbn.violinplot(x="EDUCATION",
               y="LIMIT_BAL", hue="default.payment.next.month",
               data=data, split=True,
               palette="Set1")

#%%
