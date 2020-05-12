#!/usr/bin/env python
# coding: utf-8

# Support vector Regression Model Prediction

# In[1]:


#importing all the libariries that needed to run the prediction model
import pandas as pd
import numpy as np
#sklearn libary importing support vector regression
from sklearn.svm import SVR
#importing linear regression
from sklearn.linear_model import LinearRegression
#importing matplotlib to plot the graph
import matplotlib.pyplot as plt
#importing KneighborClassfier from sklearn
from sklearn.neighbors import KNeighborsClassifier
#Importing metrics for the accuracy
from sklearn.metrics import accuracy_score
from pandas_datareader import data as pdr


# In[2]:


cd desktop/AirlinesData


# In[3]:


#predicting the company American airlines by reading the csv. we taking a month of data to predict.
AA = pd.read_csv('AAL.csv')
AA.head(22)


# In[4]:


#we need to create the dataset of x and y which is the two data set we going to take to predict the model
#so we have to take the date and we taking the "Close" cloumn
dates = []
prices = []
AA.tail(1)


# In[5]:


# retriving the data until the last row
AA = AA.head(len(AA)-1)
AA


# In[6]:


AA_dates = AA.loc[:, 'Date']
#Getting  all of the rows from the Close Column
AA_open = AA.loc[:, 'Close']


# In[7]:


#building  the independent data set for x
for date in AA_dates:
 dates.append( [int(date.split('-')[2])])
  
#building the dependent data set for y
for Close_price in AA_open:
  prices.append(float(Close_price))
#printing all the dates 
print(dates)


# In[8]:


#prediction model to build the relationship between x and y to do a prediction
def predict_prices(dates, prices, x):
  
  #Creating the  Support Vector Regression model
  svr_lin = SVR(kernel='linear', C= 1e3)
  svr_poly= SVR(kernel='poly', C=1e3, degree=2)
  svr_rbf = SVR(kernel='rbf', C=1e3, gamma='auto')
  
  #TRAINING THE DATA SET TO GET THE MODEL OF PREDICTION 
  svr_lin.fit(dates,prices)
  svr_poly.fit(dates,prices)
  svr_rbf.fit(dates,prices)
  
  #creating rhe model of LR
  lin_reg = LinearRegression()
  #Trainning the model
  lin_reg.fit(dates,prices)
  
  #Plot the models on a graph to see which has the best fit
  plt.scatter(dates, prices, color='black', label='Data')
  plt.plot(dates, svr_rbf.predict(dates), color='red', label='SVR RBF')
  plt.plot(dates, svr_poly.predict(dates), color='blue', label='SVR Poly')
  plt.plot(dates, svr_lin.predict(dates), color='green', label='SVR Linear')
  plt.plot(dates, lin_reg.predict(dates), color='orange', label='Linear Reg')
  plt.xlabel('Days')
  plt.ylabel('Price')
  plt.title('SVR MODEL')
  plt.legend()
  plt.show()
  
  return svr_rbf.predict(x)[0], svr_lin.predict(x)[0],svr_poly.predict(x)[0],lin_reg.predict(x)[0]


# In[9]:


#Predict the price of American Airlines on day 
predicted_price = predict_prices(dates, prices, [[16]])
print(predicted_price)


# #Using KNN algorithm

# In[10]:


#getting the data set for American Airlines
AA


# In[36]:


#Getting open-close and high-low to get the value
AA['Open-Close'] = AA.Open -AA.Close
AA['High-Low'] = AA.High -AA.Low
AA = AA.dropna()
X= AA[['Open-Close', 'High-Low']]
X.head()


# In[37]:


Y= np.where(AA['Close'].shift(-1)>AA['Close'],1,-1)


# In[38]:


#splitting the data and trainning the data with x and Y variables
#splitting the 80% of the data
split_percentage = 0.8
split = int(split_percentage*len(AA))

X_train = X[:split]
Y_train = Y[:split]

X_test = X[:split]
Y_test = Y[:split]


# In[39]:


#k = 12, then the object is simply assigned to the class of that nearest neighbor.
knn = KNeighborsClassifier(n_neighbors=12)
#Fitting the data for x and y VARIABLE
knn.fit(X_train, Y_train)
#trainnig the accuracy train and and test data
accuracy_train = accuracy_score(Y_train, knn.predict(X_train))
accuracy_test = accuracy_score(Y_test, knn.predict(X_test))


# In[40]:


#printing out the tran and test results for the model
print(accuracy_train)
print(accuracy_test)


# In[ ]:




