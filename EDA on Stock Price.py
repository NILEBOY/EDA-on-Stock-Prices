#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

data = pd.read_csv("Stock_Price_Dataset.csv")


# In[2]:


data.head()


# In[3]:


data.info()


# In[15]:


#drop nan rows
data = data.dropna()


# In[16]:


# Convert the date column to datetime format
data['date'] = pd.to_datetime(data['date'])


# In[17]:


# Check for duplicates
print(data.duplicated().sum())


# In[19]:


# remove any duplicate rows
data.drop_duplicates(keep=False, inplace=True)


# In[20]:


# Convert the date column to datetime format
data['date'] = pd.to_datetime(data['date'])


# In[21]:


# Rename columns
data.rename(columns={'open_value': 'Open', 'high_value': 'High', 'low_value': 'Low', 'last_value': 'Close', 'change_prev_close_percentage': 'Change_Percentage'}, inplace=True)


# In[22]:


# Handle outliers
q1 = data['Close'].quantile(0.25)
q3 = data['Close'].quantile(0.75)
iqr = q3 - q1
upper_bound = q3 + 1.5 * iqr
data = data[data['Close'] <= upper_bound]


# In[25]:


#standardize data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data[['Open', 'High', 'Low', 'Close', 'Change_Percentage', 'turnover']] = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close', 'Change_Percentage', 'turnover']])


# In[26]:


#import libraries first
import matplotlib.pyplot as plt
import seaborn as sns


# In[28]:


# Line chart of closing stock price over time
plt.figure(figsize=(10, 6))
sns.lineplot(x='date', y='Close', data=data)
plt.title('Closing Stock Price Over Time')
plt.xlabel('Date')
plt.ylabel('Closing Stock Price')
plt.show()


# In[32]:


data['year'] = data['date'].dt.year
sns.boxplot(x='year', y='Close', data=data)
plt.title('Closing Stock Prices by Year')
plt.xlabel('Year')
plt.ylabel('Closing Stock Price')
plt.show()


# In[33]:


# Create a heatmap of the correlation between stock prices
corr = data[['Open', 'High', 'Low', 'Close']].corr()
plt.figure(figsize=(8,8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Between Stock Prices')
plt.show()


# In[35]:


plt.figure(figsize=(10, 6))
sns.histplot(data['Close'], kde=True)
plt.title('Distribution of Closing Stock Price')
plt.xlabel('Closing Stock Price')
plt.ylabel('Frequency')
plt.show()


# In[40]:


daily_returns = data['Close'].pct_change()

# Create a line chart of the daily returns
plt.plot(daily_returns.index, daily_returns.values)
plt.title('Daily Returns')
plt.xlabel('Date')
plt.ylabel('Daily Return')
plt.show()
#This will create a line chart showing the daily returns over time.


# In[42]:


# Create a combination plot of stock prices and Turnover
plt.figure(figsize=(12,6))
sns.lineplot(x='date', y='Close', data=data, color='b')
sns.lineplot(x='date', y='turnover', data=data, color='g', alpha=0.5)
plt.title('Close Stock Price with Turnover')
plt.xlabel('Year')
plt.ylabel('Price/Turnover')
plt.legend(['Closing Price', 'Turnover'])
plt.show()


# In[43]:


# Create a histogram of the daily returns
plt.figure(figsize=(12,6))
sns.histplot(data['Close'].pct_change().dropna(), bins=100, kde=True)
plt.title('Daily Returns')
plt.xlabel('Daily Return')
plt.ylabel('Frequency')
plt.show()


# In[ ]:




