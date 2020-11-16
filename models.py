#loading the dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
train=pd.read_csv("train_data.csv")
test=pd.read_csv("Test_data.csv")

# AUTO-REGRESSIVE MODEL
from statsmodels.tsa.ar_model import AutoReg
# fit model
model = AutoReg(train["Close_Value"], lags=1)
model_fit = model.fit()
# make prediction
yhat3 = model_fit.predict(1, 2548)
print(yhat3)
print(model_fit.summary())
print("BIC: ",model_fit.bic)
mse=np.square(np.subtract(test["Close_Value"],yhat3)).mean()
print ("MSE: ",mse)
#plot
x=list(range(len(test)))
plt.plot(x,test["Close_Value"],c='blue')
plt.plot(x,yhat3,c='green')
plt.legend()
plt.show()

# MOVING AVERAGE
from statsmodels.tsa.arima_model import ARMA
# fit model
model = ARMA(train["Close_Value"], order=(0, 1))
model_fit = model.fit(disp=False)
# make prediction
yhat = model_fit.predict(1,2548)
print(yhat)
print(model_fit.summary())
print("BIC: ",model_fit.bic)
mse=np.square(np.subtract(test["Close_Value"],yhat)).mean()
print ("MSE: ",mse)
#plot
plt.plot(x,test["Close_Value"],c='blue')
plt.plot(x,yhat,c='green')
plt.legend()
plt.show()

# AUTO-REGRESSIVE MOVING AVERAGE
from statsmodels.tsa.arima_model import ARMA
# fit model
model = ARMA(train["Close_Value"], order=(0, 1))
model_fit = model.fit(disp=False)
# make prediction
yhat = model_fit.predict(1,len(test))
print(yhat)
print(model_fit.summary())
print("BIC: ",model_fit.bic)
mse=np.square(np.subtract(test["Close_Value"],yhat)).mean()
print ("MSE: ",mse)
#plot
plt.plot(x,test["Close_Value"],c='blue')
plt.plot(x,yhat,c='green')
plt.legend()
plt.show()

# AUTO-REGRESSIVE INTEGRATED MOVING AVERAGE
from statsmodels.tsa.arima_model import ARIMA
# fit model
model = ARIMA(train["Close_Value"], order=(1, 1, 1))
model_fit = model.fit(disp=False)
# make prediction
yhat1 = model_fit.predict(1, 2548, typ='levels')
print(model_fit.summary())
print("BIC: ",model_fit.bic)
mse=np.square(np.subtract(test["Close_Value"],yhat1)).mean()
print ("MSE: ",mse)
#plot
plt.plot(x,test["Close_Value"],c='blue')
plt.plot(x,yhat1,c='green')
plt.legend()
plt.show()

# SEASONAL AUTO-REGRESSIVE INTEGRATED MOVING AVERAGE
from statsmodels.tsa.statespace.sarimax import SARIMAX
# fit model
model = SARIMAX(train["Close_Value"], order=(1, 1, 1), seasonal_order=(1,1,1,2))
model_fit = model.fit(disp=False)
# make prediction
yhat2 = model_fit.predict(1, 2548)
print(yhat2)
print(model_fit.summary())
print("BIC: ",model_fit.bic)
mse=np.square(np.subtract(test["Close_Value"],yhat2)).mean()
print ("MSE: ",mse)
#plot
x=list(range(len(test)))
plt.plot(x,test["Close_Value"],c='blue')
plt.plot(x,yhat2,c='green')
plt.legend()
plt.show()


#SIMPLE EXPONENTIAL SMOOTHING
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
# fit model
model = SimpleExpSmoothing(train["Close_Value"])
model_fit = model.fit()
# make prediction
yhat = model_fit.predict(1, 2548)
print(yhat)
print(model_fit.summary())
print("BIC: ",model_fit.bic)
mse=np.square(np.subtract(test["Close_Value"],yhat)).mean()
print ("MSE: ",mse)
#plot
plt.plot(x,test["Close_Value"],c='blue')
plt.plot(x,yhat,c='green')
plt.legend()
plt.show()

# HOLT WINTER'S EXPONENTIAL SMOOTHING
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from random import random
# contrived dataset
#data = [x + random() for x in range(1, 100)]
# fit model
model = ExponentialSmoothing(train["Close_Value"])
model_fit = model.fit()
# make prediction
yhat = model_fit.predict(1, 2548)
print(yhat)
print(model_fit.summary())
print("BIC: ",model_fit.bic)
mse=np.square(np.subtract(test["Close_Value"],yhat)).mean()
print ("MSE: ",mse)
#plot
plt.plot(x,test["Close_Value"],c='blue')
plt.plot(x,yhat,c='green')
plt.legend()
plt.show()