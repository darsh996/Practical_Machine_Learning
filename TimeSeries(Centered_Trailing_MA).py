#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 08:25:32 2023

@author: darshmac
"""

import pandas as pd
import numpy as np
from numpy import sqrt
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
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

plt.plot(y_train,label='Train')
plt.plot(y_test,label='Test')
plt.plot(MA_fcast,label = 'Moving Average Forecast')
plt.legend(loc = 'best')
plt.show()

rms = sqrt(mean_squared_error(y_test, MA_series))
print(rms)


###########SES
from statsmodels.tsa.api import SimpleExpSmoothing

alpha = 0.5
ses = SimpleExpSmoothing(y_train)
fit1 = ses.fit(smoothing_level=alpha)
fcast1 = fit1.forecast(len(y_test))

#plot
y_train.plot(color="Blue", label='train')
y_test.plot(color='pink',label='Test')
fcast1.plot(color='purple',label='Forecast')
plt.legend(loc='best')
plt.show()


plt.legend(loc = 'best')
plt.show()




