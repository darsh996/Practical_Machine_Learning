#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 08:25:32 2023

@author: darshmac
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import os
os.chdir("/Users/darshmac/Documents/cdac/For DBDA/Advance Analytics/Datasets")

df = pd.read_csv("monthly-milk-production-pounds-p.csv")
df.head()

y = df['Milk']

y_train = df['Milk'][:-12]
y_test = df['Milk'][-12:]

###########trailing MA

fcast = y_train.rolling(3,center =False).mean()

MA = fcast.iloc[-1]
MA_series = pd.Series(MA.repeat(len(y_test)))
MA_fcast = pd.concat([fcast,MA_series],ignore_index=True)








