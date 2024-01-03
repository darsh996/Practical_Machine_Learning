import pandas as pd
import matplotlib.pyplot as plt
from numpy import sqrt
from sklearn.metrics import mean_squared_error as mse

df = pd.read_csv("monthly-milk-production-pounds-p.csv")
df.head()

y = df['Milk']
#### Centered MA
fcast = y.rolling(3,center=True).mean()
plt.plot(y, label='Data')
plt.plot(fcast, label='Centered Moving Average')
plt.legend(loc='best')
plt.show()

y_train = df['Milk'][:-12]
y_test = df['Milk'][-12:]

###### Trailing MA ############
fcast = y_train.rolling(6,center=False).mean()
MA = fcast.iloc[-1]
MA_series = pd.Series(MA.repeat(len(y_test)))
MA_fcast = pd.concat([fcast,MA_series],ignore_index=True)
plt.plot(y_train, label='Train')
plt.plot(y_test, label='Test')
plt.plot(MA_fcast, label='Moving Average Forecast')
plt.legend(loc='best')
plt.show()

rms = sqrt(mse(y_test, MA_series))
print(rms)

################# SES #######################
from statsmodels.tsa.api import SimpleExpSmoothing 

alpha = 0.05
ses = SimpleExpSmoothing(y_train)
fit1 = ses.fit(smoothing_level=alpha)
fcast1 = fit1.forecast(len(y_test))

# plot
y_train.plot(color="blue", label='Train')
y_test.plot(color="pink", label='Test')
fcast1.plot(color="purple", label='Forecast')
plt.legend(loc='best')
plt.show()

print("RMSE =",sqrt(mse(y_test, fcast1)))

########## Holt's Linear Method
alpha = 0.3
beta = 0.02
### Linear Trend
from statsmodels.tsa.api import Holt
holt = Holt(y_train)
fit1 = holt.fit(smoothing_level=alpha, 
                smoothing_trend=beta)
# fit1 = holt.fit() 
# automatically fits with best alpha & beta 
# for train set 
fcast1 = fit1.forecast(len(y_test))

# plot
y_train.plot(color="blue", label='Train')
y_test.plot(color="pink", label='Test')
fcast1.plot(color="purple", label='Forecast')
plt.legend(loc='best')
plt.show()

print("RMSE =",sqrt(mse(y_test, fcast1)))

########## Holt's Exponential Method
alpha = 0.3
beta = 0.02
### Expo Trend
from statsmodels.tsa.api import Holt
holt = Holt(y_train, exponential=True)
fit1 = holt.fit(smoothing_level=alpha, 
                smoothing_trend=beta)
# fit1 = holt.fit() 
# automatically fits with best alpha & beta 
# for train set 
fcast1 = fit1.forecast(len(y_test))

# plot
y_train.plot(color="blue", label='Train')
y_test.plot(color="pink", label='Test')
fcast1.plot(color="purple", label='Forecast')
plt.legend(loc='best')
plt.show()

print("RMSE =",sqrt(mse(y_test, fcast1)))

########## Damped Exponential Method
alpha = 0.3
beta = 0.02
phi = 0.2
### Expo Trend with damping
from statsmodels.tsa.api import Holt
holt = Holt(y_train, exponential=True, damped_trend=True)
fit1 = holt.fit(smoothing_level=alpha, 
                smoothing_trend=beta, damping_trend=phi)
# fit1 = holt.fit() 
# automatically fits with best alpha & beta 
# for train set 
fcast1 = fit1.forecast(len(y_test))

# plot
y_train.plot(color="blue", label='Train')
y_test.plot(color="pink", label='Test')
fcast1.plot(color="purple", label='Forecast')
plt.legend(loc='best')
plt.show()

print("RMSE =",sqrt(mse(y_test, fcast1)))


# Holt-Winters' Method

########### Additive #####################
from statsmodels.tsa.api import ExponentialSmoothing
alpha = 0.8
beta = 0.02
gamma = 0.1
hw_add = ExponentialSmoothing(y_train, seasonal_periods=12, 
                            trend='add', seasonal='add')
fit1 = hw_add.fit(smoothing_level=alpha, 
                    smoothing_trend=beta,
                    smoothing_seasonal=gamma)

# fit1 = hw_add.fit()
# automatically fits with best alpha , beta , gamma 
# for train set 
fcast1 = fit1.forecast(len(y_test))

print("RMSE =",sqrt(mse(y_test, fcast1)))

# plot
y_train.plot(color="blue", label='Train')
y_test.plot(color="pink", label='Test')
fcast1.plot(color="purple", label='Forecast')
plt.legend(loc='best')
plt.show()


########### Multiplicative #####################
alpha = 0.8
beta = 0.02
gamma = 0.1
hw_mul = ExponentialSmoothing(y_train, seasonal_periods=12, 
                            trend='add', seasonal='mul')
fit1 = hw_mul.fit(smoothing_level=alpha, 
                    smoothing_trend=beta,
                    smoothing_seasonal=gamma)
fcast1 = fit1.forecast(len(y_test))

print("RMSE =",sqrt(mse(y_test, fcast1)))

# plot
y_train.plot(color="blue", label='Train')
y_test.plot(color="pink", label='Test')
fcast1.plot(color="purple", label='Forecast')
plt.legend(loc='best')
plt.show()
