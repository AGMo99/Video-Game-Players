# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 05:16:10 2021

@author: AG
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
import matplotlib.pyplot as plt

#Upload Data and Check

df = pd.read_csv(r'c:\users\ag\desktop\Projects\project 10\Valve_Player_Data.csv')

df.head()
df.info(null_counts=True)
df.isna().sum()

df['Gain'].isnull().sum()
df['Percent_Gain'].isnull().values.any()

df = df.dropna()

plt.figure(figsize=(20, 15))
plt.plot(df['Avg_players'], label = 'Average Players')
plt.plot(df['Gain'], label = 'Gain')
plt.legend()
plt.show()

#Simple Analysis

dfg = df.groupby('Date')['Peak_Players'].sum()

dfg.describe()

dfg.plot()

dfd = df.groupby('Date')['Game_Name'].count()

dfd.describe()

dfd.plot()

dfp = df.groupby('Game_Name')['Percent_Gain'].count()

dfp.plot()

#Standardize and Normalize Data
sddf = preprocessing.scale(df[['Avg_players', 'Gain', 'Peak_Players']])
scaler = preprocessing.MinMaxScaler()
nordf = scaler.fit_transform(sddf)

#Model

x = nordf[:, 0]
y = df['Game_Name']

x = np.reshape(nordf, (-1, 1))

random_state = 5000
y_pred = KMeans(n_clusters= 1000, random_state=random_state).fit_predict(x)

plt.figure(figsize=(20, 20))
plt.scatter(x, y_pred, c = y_pred)
plt.show()