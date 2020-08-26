#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels import regression
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import requests
from pprint import pprint
import json
import datetime


# In[ ]:


Companies = {'Apple':'AAPL','Amazon':'AMZN','Facebook':'FB','Netflix':'NFLX','Tesla':'TSLA'}
print(Companies)


# In[ ]:


Stock = input("Which stock did the class choose?:")


# In[ ]:


data = pd.read_csv(f'{Stock}.csv',index_col="Date",parse_dates=True)
data.shape


# In[ ]:


data.head()


# In[ ]:


data['Close'].plot(figsize=(18,8))
plt.title(f'{Stock} Preformance Since 2016')
plt.xlabel("Date")
plt.ylabel("Close Price($)")
#plt.savefig(f'{Stock}_since2016.png')


# In[ ]:


training_set = data['Close']
training_set = pd.DataFrame(training_set)


# In[ ]:


#to normalize our data to values between 0 and 1
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_set)


# In[ ]:


#we'll use those prior 90 days to train (thought process of 1 economic quarter)

X_train = []
y_train = []
for i in range(90,1167):
    X_train.append(training_set_scaled[i-90:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# In[ ]:


regressor = Sequential()
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1],1)))
regressor.add(Dropout(0.2))#reduce overfitting

regressor.add(LSTM(units = 50, return_sequences = True))#add a layer
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))#add another layer
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))


# In[ ]:


#Adam
#(theta sub(t +1)) = (theta sub(t)) - [ (eta)/sqrt(v sub(t) + epilson) * m sub(t)]
#include weight, but don't let outliers overweigh the model

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
regressor.fit(X_train, y_train, epochs = 10, batch_size = 64)


# In[ ]:


days_out = 90
data['Prediction'] = data[['Close']].shift(-days_out)
data.tail(10)


# In[ ]:


X = np.array(data.drop(['Prediction'],1))
X = X[:-days_out]
#print(X)

y = np.array(data['Prediction'])
y = y[:-days_out]
#print(y)


# In[ ]:


#train on 80% test on 20%
X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
lr = LinearRegression()
lr.fit(X_train, y_train)


# In[ ]:


confidence = lr.score(x_test, y_test)
print(f'lr confidence: {confidence}')


# In[ ]:


x_forecast = np.array(data.drop(['Prediction'],1))[-days_out:]
#print(x_forecast)
lr_prediction = lr.predict(x_forecast)
print(lr_prediction)


# In[ ]:


#iterate through data table, find values will labeled NaN, then insert prediction values found above
import math
counter = 0
for i in range(len(data)):
    print(math.isnan(data['Prediction'][i]))
    
    if math.isnan(data['Prediction'][i]):
        data['Prediction'][i] = lr_prediction[counter]
        #print(lr_prediction[counter])
        #print(counter)
        counter+=1
        #print(counter)
    else:
        pass


# In[ ]:


data.tail()


# In[ ]:


#plotting the predictions vs actual close price
data2 = data
data2[['Close','Prediction']].plot(figsize=(18,8))

#modify ylim to class stock pick
data2[['Close','Prediction']].plot(figsize=(18,8), xlim=('2020-05-15','2020-08-20'), ylim=(275,550))

plt.title(f'{Stock} Preformance in the last Month')
plt.xlabel("Date")
plt.ylabel("Close Price($)")


# We are going to implement a trading strategy following momentum using Dual Moving Averages, also known as Moving Average Crossover. It is a popular technical trading strategy for investors; who do not have the time to Day Trade, find entry and exit positions with a security. "Day Trading is simply buying and selling a single security within a single day" (Kuepper, 2020). There are infinite combinations using moving averages to find change in direction. For demonstration purposes, I will just use 2 simple moving averages.

# In[ ]:


#how manys days into the past should we look to make the moving averages (I chose a couple numbers from fibonacci seq.)
# 8 days / 34 days
#make a column where the computer confirms if the moving averages cross eachother by making it a -1,0, or 1
signal = pd.DataFrame(index=data.index)
signal['signal'] = 0.0

signal['Fast_SMA'] = data['Close'].rolling(window=8, min_periods=1, center=False).mean()
signal['Slow_SMA'] = data['Close'].rolling(window=34, min_periods=1, center=False).mean()

#Generate signals when moving averages cross, have to keep the window the same bc it is a boolean
signal['signal'][8:] = np.where(signal['Fast_SMA'][8:] > signal['Slow_SMA'][8:], 1.0, 0.0)   

# If new row is different than the previous row, make signal change
signal['positions'] = signal['signal'].diff()
#signal


# In[ ]:


#when the fast moving average moves above the slow moving average, we'll initiate a buy signal
buy = signal.loc[signal.positions == 1.0]
buy['Buy'] = str('Buy')
buy.to_csv(f'{Stock}buyDates.csv')
buy.head()


# In[ ]:


#when the fast moving average moves below the slow moving average, we'll initiate a sell signal
sell = signal.loc[signal.positions == -1.0]
sell['Sell'] = str('Sell')
sell.to_csv(f'{Stock}sellDates.csv')
#sell


# In[ ]:


#combine buys and sells into one seperate dataframe to filter through later for news
buysandsells = pd.concat([buy,sell], axis=1)
buysandsells


# In[ ]:


buysandsells2 = buysandsells.reset_index('Date')
buysandsells2


# In[ ]:


#get the last signal date and let's take a look at the news trending on that date
df5 = str(buysandsells2.iloc[-1,:]['Date']).split(" ")
last_signal = df5[0]
last_signal
#print(f'The last trade signal was on: {last_signal}')


# In[ ]:


last_signal_news = requests.get(f'https://finnhub.io/api/v1/company-news?symbol={Stock}&from={last_signal}&to={last_signal}&token=')
pprint(last_signal_news.json())


# In[ ]:


json_response = last_signal_news.json()
for i in range(len(json_response)):
    yellow = datetime.datetime.fromtimestamp(int(json_response[i]['datetime'])).strftime('%Y-%m-%d %H:%M:%S')
    print(f'({Stock})', yellow)
    print(json_response[i]['headline'])
    print(json_response[i]['url'])
    print()


# In[ ]:


#put it all together to plot
combo = pd.concat([data, buysandsells], axis=1)
#combo.to_csv('combo.csv')
combo


# In[ ]:


fig = plt.figure(figsize=(18,5))
ax1 = fig.add_subplot(title=f'{Stock}\'s Preformace since 2016', ylabel='Price (USD)')
data['Close'].plot(ax=ax1)
data[['Close']].plot(ax=ax1, color='black', lw=2)
signal[['Fast_SMA', 'Slow_SMA']].plot(ax=ax1)

#buy signals with green triangles
buy = ax1.plot(signal.loc[signal.positions == 1.0].index, signal.Fast_SMA[signal.positions == 1.0],'g^', markersize=10, color='g')
buy
#sell signals with red triangles
sell = ax1.plot(signal.loc[signal.positions == -1.0].index, signal.Fast_SMA[signal.positions == -1.0],'v', markersize=10, color='red')
sell

#plt.savefig(f'{Stock}_entryexits.png')
plt


# Time to put it to the test against the analysts' recommendations!

# In[ ]:


playground_money = float(input("How much money would you like to start with?:"))


# In[ ]:


#make a new dataframe to find out total percentage returns to compare to analyst returns

positions = pd.DataFrame(index=signal.index).fillna(0.0)
positions[f'Shares of {Stock}'] = (playground_money/(combo['Close']))*signal['signal']   
portfolio = positions.multiply(data['Close'], axis=0)
pos_diff = positions.diff()
portfolio['holdings'] = (positions.multiply(data['Close'], axis=0)).sum(axis=1)
portfolio['cash'] = playground_money - (pos_diff.multiply(data['Close'], axis=0)).sum(axis=1).cumsum()   
portfolio['total'] = portfolio['cash'] + portfolio['holdings']
#print(portfolio)
print("---------------------")

purple = portfolio['cash'].iloc[-1]
total_return = (purple - playground_money)/playground_money*100
print(f'Starting Capital = ${playground_money}')
print(f'Ending Capital = ${round(purple,2)}')
print(f'Net percentage return: {round(total_return,2)}%')
print("---------------------")


# In[ ]:


#to see our buy and sell signals
number_of_shares = []
total = []

for i in range(len(combo)):
    if combo['Buy'][i] == 'Buy':
        number_of_shares = playground_money/(combo['Close'][i])
        a = combo['Close'][i]
        print(f'Bought at ${round(a,2)} a share.')
        
    elif combo['Sell'][i] == 'Sell':
        sold = combo['Close'][i]
        total = int(number_of_shares)*(combo['Close'][i])
        
        net_returns = round(total - playground_money,2)
        percent_return = round(net_returns/playground_money,2)
    
        print(f'Sold at ${round(sold,2)} a share.')
        print('-----------------')
        print()
        
    else:
        pass


# In[ ]:





# If we wanted to explore a decision tree method and test that accuracy. How does a Decision Tree Regressor work?
# "Given a data point you run it through the entirely tree asking True/False questions up until it reaches a leaf node. The final prediction is the average of the value of the dependent variable in that leaf node" (Drakos, 2019).

# In[ ]:


econ_quarter = 90
data['Prediction'] = data[['Close']].shift(-econ_quarter)
data.tail()


# In[ ]:


X = np.array(data.drop(['Prediction'],1))[:-econ_quarter]
print(X)


# In[ ]:


y = np.array(data['Prediction'])[:-econ_quarter]
print(y)


# In[ ]:


X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[ ]:


dtree = DecisionTreeRegressor().fit(X_train, y_train)
lin_reg = LinearRegression().fit(X_train, y_train)


# In[ ]:


x_future = data.drop(['Prediction'],1)[:-econ_quarter]
x_future = x_future.tail(econ_quarter)
x_future = np.array(x_future)
x_future


# In[ ]:


tree_prediction = dtree.predict(x_future)
print(tree_prediction)


# In[ ]:


actual_close = data[X.shape[0]:]
#actual_close
actual_close['Predictions'] = tree_prediction
plt.figure(figsize=(18,8))
plt.plot(data['Close'])
plt.plot(actual_close[['Close','Predictions']])

plt.title('Predictions')
plt.xlabel('Date')
plt.ylabel('Close ($)')
plt.legend(['Original','Actual_Close','Predictions'])
plt


# In[ ]:


actual_close2 = data[X.shape[0]:]
#actual_close2
actual_close2['Prediction'] = tree_prediction
actual_close2


actual_close2[['Close','Prediction']].plot(figsize=(18,8), xlim=('2020-04-15','2020-08-20'), ylim=(220,600))

#plt2.legend(['Original','Actual','Predictions'])
#plt2.title('Predictions')
#plt2.xlabel('Date')
#plt2.ylabel('Close ($)')


# In[ ]:


#predictions and actual close with 8, 34, and 200 moving averages

data['Close: 8 Day MA'] = data['Close'].rolling(window=8).mean()
data['Close: 34 Day MA'] = data['Close'].rolling(window=34).mean()
data['Close: 200 Day MA'] = data['Close'].rolling(window=200).mean()
plt1 = data[['Close','Prediction','Close: 8 Day MA','Close: 34 Day MA','Close: 200 Day MA']].plot(figsize=(18,8))
plt2 = data[['Close','Prediction','Close: 8 Day MA','Close: 34 Day MA','Close: 200 Day MA']].plot(figsize=(18,8), xlim=("2019-11-30","2020-08-20"), ylim=(230,600))
plt1
plt2


# In[ ]:


#import ResearchTeam's suggestions
import matplotlib.image as mping
img = mping.imread(f'{Stock}_RT.png')
plt.figure(figsize=(20,20))
plt.imshow(img)


# In[ ]:


References

Brownlee, J. (August 20, 2020). Time Series Prediction with LSTM Recurrent Neural Networks in Python with Keras. Retrived from https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/

Drakos, G. (May 23, 2019). Decision Tree Regressor explained in depth. Retrieved from https://gdcoder.com/decision-tree-regressor-explained-in-depth/

Hayes, A. (Mar 26, 2020). Simple Moving Average (SMA). Retreived from https://www.investopedia.com/terms/s/sma.asp  

Kenton, W. (Feb 6, 2020). Black Scholes Model. Retrieved from https://www.investopedia.com/terms/b/blackscholes.asp 

Kenton, W. (May 17, 2020). Merton Model Defintion. Retrived from https://www.investopedia.com/terms/m/mertonmodel.asp 

Kuepper, J. (Aug 11,2020). Day Trading: An Introduction. Retrieved from https://www.investopedia.com/articles/trading/05/011705.asp

(n.d.) Retrived from https://finnhub.io/docs/api#company-news

