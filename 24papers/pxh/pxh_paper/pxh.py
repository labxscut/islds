#!/usr/bin/env python
# coding: utf-8

# # Predicting Stock Prices and Asset Bubbles Using Deep Learning: A Case Study of Nvidia 

# ## Importing Packages

# In[417]:


# Import necessary libraries
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import seaborn as sns

import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.layers import Dense, Conv1D, LSTM, Lambda, Reshape, RNN, LSTMCell, Concatenate, Embedding, Flatten
from tensorflow.keras.layers import Dropout, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import joblib
import string


import tweepy
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import statsmodels.api as sm

import matplotlib.ticker as ticker
import warnings
warnings.filterwarnings('ignore')


# In[232]:


# Set default figure size for visualizations
plt.rcParams['figure.figsize'] = (10, 4)


# In[233]:


# Set random seed values for TensorFlow and NumPy libraries
tf.random.set_seed(42)  
np.random.seed(42) 


# In[234]:


cwd = os.getcwd()


# ## Data preprocessing 

# ### NVIDIA Stock Data preprocessing 

# We will use NVIDIA stock data, and the dataset contains a total of seven columns as shown below.
# - Date: The period in which the Open, High, Low, Close, and Volume (OHLCV) of the security price are discussed.
# - Volume: The total number of shares traded during the security period
# - High: The highest price at which a stock has traded during a period of time
# - Low: The lowest price at which a stock has traded during a period of time
# - Open: The price at which a stock starts trading at the opening of a particular day 6.Closing: The price agreed upon by traders after all the action of the day, and 7.Adjusted Closing: While the closing price refers only to the
# - Cost of the stock at the end of the day, the adjusted closing price takes into account dividends, stock splits, and new stock issuances.
# 
# The primary target is the closing price, as it represents the last trading price of the stock during the regular trading session.

# In[372]:


# Assuming you have two datasets: historical_stock_data.csv 
historical_data = pd.read_csv('NvidiaStockPrice.csv')
historical_data.head()


# In[373]:


# check end-date of data
historical_data.tail()


# This means that our data starts on January 22, 1999, and ends on August 23, 2024, with data collected every day.

# In[374]:


historical_data.shape


# In[375]:


# Check Missing Data
historical_data.isna().sum()


# In[376]:


# Check Categorical values
historical_data.dtypes


# In[377]:


# Convert 'Date' column to datetime data type
historical_data['Date'] = pd.to_datetime(historical_data['Date'])


# In[378]:


# Data Resampling
# Group the data by month and count the number of unique days per month
days_per_month = historical_data.groupby(historical_data['Date'].dt.to_period('M'))['Date'].nunique()

# Convert series to dataframe
days_per_month = pd.DataFrame(days_per_month)

# Filter results to show only months with less than 15 days
days_per_month = days_per_month[days_per_month['Date'] < 15]

print(days_per_month)


# In[379]:


# Drop rows where the year is 1999
historical_data = historical_data[historical_data['Date'].dt.year != 1999]

#  resetting the DataFrame index
historical_data.reset_index(inplace = True, drop = True)

historical_data.head(5)


# In[380]:


# Visualize closing prices
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(historical_data['Close'])
ax.set_xlabel('Date')
ax.set_ylabel('Closing price (USD)')
ax.set_xlim(0, 6440)
fig.autofmt_xdate()
plt.tight_layout()


# The chart indicates a clear upward trend, especially in the latter part of the series. This trend suggests a non-stationary time series.The sharp peaks and dips could indicate outliers or noise, which are common in financial markets due to sudden news events, macroeconomic announcements, or changes in investor sentiment. Preprocessing the data to handle outliers and noise is necessary.

# ### Financial statement data preprocessing 

# These indicators are derived from the company's financial statements, providing insights into financial health and performance indicators. A total of nearly four years of financial data were collected, and the data comes from Yahoo Finance

# #### Balance_data

# In[381]:


# Check Categorical values
Balance_data.dtypes


# In[358]:


# Read in the data and set the index
Balance_data = pd.read_csv('Balance_data.csv', header=0)
Balance_data.set_index(list(Balance_data.columns)[0], inplace=True)

# Replace invalid and missing values with NaN
Balance_data.replace(['-', '--', '---', '----', '/', 'NaN', ''], np.nan, inplace=True)

# Convert comma-separated values to float
comma_cols = Balance_data.select_dtypes(include=['object']).columns
Balance_data[comma_cols] = Balance_data[comma_cols].apply(lambda x: x.str.replace(',', '').astype(float))

# Transpose the data frame
Balance_data_transposed = Balance_data.T
Balance_data_transposed.head()


# #### Income_data

# In[359]:


# Check Categorical values
Income_data.dtypes


# In[360]:


# View Income sheet data
# All numbers in thousands
Income_data= pd.read_csv('Income_data.csv', header=0)
Income_data.set_index(list(Income_data.columns)[0], inplace=True)
Income_data.replace(['-','--', '---', '----'], np.nan, inplace=True)

# Convert columns with commas to float
comma_cols = Income_data.select_dtypes(include=object).columns
Income_data[comma_cols] = Income_data[comma_cols].apply(lambda x: x.str.replace(',', '').astype(float))

# Transpose the dataframe
Income_data_transposed = Income_data.T
Income_data_transposed.head()


# In[249]:


# Check Categorical values
Income_data_transposed.dtypes


# #### Cash flow sheet data

# In[250]:


# Check Categorical values
Cash_flow_data.dtypes


# In[251]:


# View Cash Flow sheet data
# All numbers in thousands
Cash_flow_data = pd.read_csv('Cash_flow_data.csv', header=0)
Cash_flow_data.set_index(list(Cash_flow_data.columns)[0], inplace=True)
Cash_flow_data.replace(['-','--', '---', '----'], np.nan, inplace=True)

# Convert columns with commas to float
comma_cols = Cash_flow_data.select_dtypes(include=object).columns
Cash_flow_data[comma_cols] = Cash_flow_data[comma_cols].apply(lambda x: x.str.replace(',', '').astype(float))

# Transpose the dataframe
Cash_flow_data_transposed = Cash_flow_data.T
Cash_flow_data_transposed.head()


# In[252]:


# Check Categorical values
Cash_flow_data_transposed.dtypes


# ### News Data preprocessing 

# This dataset contains selected financial news articles related to NVIDIA from Yahoo and X (Twitter) between August 2015 and June 2024. The collection and compilation of this dataset focuses on capturing key financial news, market performance insights, strategic business initiatives, and technological advancements that impact these companies. The data covers a range of release dates, providing a temporal view of major events and their potential impact on the market and company valuations.
# 
# The financial news data is fed into a hybrid deep learning model to extract three main features: sentiment analysis, news volume, and event impact. These features are valuable because they reflect market sentiment, investor behavior, and major events that may affect stock prices.

# In[169]:


nvidia_sentimental_analisis= pd.read_csv('nvidia_sentimental_analisis.csv')


# In[170]:


print(nvidia_news_analisis.head())
print(nvidia_news_analisis.info())
print(nvidia_news_analisis['sentiment_rf'].value_counts())
print(nvidia_news_analisis['sentiment_nn'].value_counts())


# ### S&P 500 Stocks

# The Standard and Poor's 500 or S&P 500 is the most famous financial benchmark in the world.
# 
# This stock market index tracks the performance of 500 large companies listed on stock exchanges in the United States. As of December 31, 2020, more than $5.4 trillion was invested in assets tied to the performance of this index.
# 
# Because the index includes multiple classes of stock of some constituent companies—for example, Alphabet's Class A (GOOGL) and Class C (GOOG)—there are actually 505 stocks in the gauge.

# In[177]:


sp500_companies= pd.read_csv('sp500_companies.csv')
sp500_index= pd.read_csv('sp500_index.csv')
sp500_stocks= pd.read_csv('sp500_stocks.csv')


# In[179]:


sp500_companies.head()


# In[178]:


sp500_index.head()


# In[180]:


sp500_stocks.head()


# In[181]:


sp500_stocks_describe = sp500_stocks.describe()
sp500_stocks_describe


# ### Historical 3-month Treasury Bill Rates (2000-2023)

# This dataset contains historical 3-month Treasury Bill rates, sourced from Yahoo Finance. The dataset spans from January 3, 2000, to December 31, 2023, and provides daily prices along with adjusted close prices and volumes. This data is crucial for financial analysts, economists, and researchers who are interested in interest rate trends and their impact on the economy.

# In[183]:


Historical_3m_treasury_bill_interest_rates= pd.read_csv('3m_treasury_bill_interest_rates.csv')


# In[184]:


Historical_3m_treasury_bill_interest_rates.head()


# In[185]:


Historical_3m_treasury_bill_interest_rates_describe = Historical_3m_treasury_bill_interest_rates.describe()
Historical_3m_treasury_bill_interest_rates_describe


# ### US_Recession data

# This dataset includes various economic indicators such as stock market performance, inflation rates, GDP, interest rates, employment data, and housing index, all of which are crucial for understanding the state of the economy. By analysing this dataset, one can gain insights into the causes and effects of past recessions in the US, which can inform investment decisions and policy-making.
# 
# There are 20 columns and 343 rows spanning 1990-04 to 2022-10

# In[187]:


US_Recession= pd.read_csv('US_Recession.csv')


# In[188]:


US_Recession.head()


# In[189]:


US_Recession_describe = US_Recession.describe()
US_Recession_describe


# ## Feature calculation

# ### Technical indicators

# These three indicators are among the most commonly used characteristic values ​​in the stock market, and they provide important information about the movement and trend of stock prices.
# 
# - The moving average (MA) shows the trend of a stock price by calculating the average of the stock price over a period of time. By comparing the averages of adjacent time periods, the change in trend and its possible trend direction can be found.
# 
# - The relative strength index (RSI) shows the strength and weakness of a stock by comparing the number of rising days and falling days of a stock over a period of time. The maximum value of the RSI indicator is 100 and the minimum value is 0. According to the high and low RSI values, the strength and weakness of the stock can be judged and whether it is overbought or oversold.
# 
# - Bollinger Bands is a technical analysis tool used to determine the high and low stock prices and the possible range of price fluctuations. Bollinger Bands consists of three curves: the middle line represents the 20-day moving average of the stock price; the upper and lower lines are composed of the distance between the middle line and one standard deviation, which are used to show the range of price fluctuations. By comparing the stock price to the position of the Bollinger Bands, it is possible to determine if the price is excessively high or low and a trend reversal is likely. This makes Bollinger Bands one of the most useful analysis tools in the stock market.

# In[321]:


# data describe
historical_data_describe = historical_data.describe()
historical_data_describe


# In[81]:


# calculate the MA using a rolling window of 20 days
historical_data['MA'] = historical_data['Close'].rolling(window=20).mean()

# calculate the RSI
delta = historical_data['Close'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
historical_data['RSI'] = 100 - (100 / (1 + rs))

# calculate the Bollinger Bands using a rolling window of 20 days
historical_data['MA20'] = historical_data['Close'].rolling(window=20).mean()
historical_data['Std20'] = historical_data['Close'].rolling(window=20).std()
historical_data['UpperBand20'] = historical_data['MA20'] + (historical_data['Std20'] * 2)
historical_data['LowerBand20'] = historical_data['MA20'] - (historical_data['Std20'] * 2)


# In[71]:


# plot the MA, RSI, and Bollinger Bands
plt.figure(figsize=(12, 8))
plt.plot(historical_data['Date'], historical_data['Close'], label='Closing Price')
plt.plot(historical_data['Date'], historical_data['MA'], label='MA(20)')
plt.legend(loc='upper left')
plt.title('Moving Average')
plt.xlabel('Date')
plt.ylabel('Price')

plt.figure(figsize=(12, 8))
plt.plot(historical_data['Date'], historical_data['RSI'])
plt.title('Relative Strength Index')
plt.xlabel('Date')
plt.ylabel('RSI')

plt.figure(figsize=(12, 8))
plt.plot(historical_data['Date'], historical_data['Close'], label='Closing Price')
plt.plot(historical_data['Date'], historical_data['MA20'], label='MA(20)')
plt.plot(historical_data['Date'], historical_data['UpperBand20'], label='Upper Band')
plt.plot(historical_data['Date'], historical_data['LowerBand20'], label='Lower Band')
plt.legend(loc='upper left')
plt.title('Bollinger Bands')
plt.xlabel('Date')
plt.ylabel('Price')

plt.show()


# ### Volume Indicators

# Volume, On Balance Volume (OBV) and Volume Weighted Average Price (VWAP) are also commonly used features in stock price prediction and asset bubble prediction:
# 
# - Volume: Stock volume refers to the number of stocks traded on an exchange during a certain period of time. When the volume rises sharply, it generally indicates that there are a lot of trading opportunities in the market, which may lead to fluctuations in asset prices and the formation of bubbles.
# 
# - On Balance Volume (OBV): OBV refers to a technical analysis tool that can be used to measure the power comparison between buyers and sellers to predict the possibility of price increases or decreases. OBV is calculated by comparing the OBV of the previous day with the trading volume of the day. If the closing price of the day is higher than the previous day, the trading volume of the day is added to the OBV, otherwise the negative number of the trading volume of the day is added to the OBV.
# 
# - Volume Weighted Average Price (VWAP): VWAP refers to the weighted average of all transaction prices calculated with trading volume as the weight. VWAP can measure the average price over a period of time and reflect the overall situation of the securities market. In stock price and asset bubble prediction, VWAP can be used to analyze the effectiveness of high-frequency trading data and systematic trading strategies.

# In[86]:


# Calculate volume change
historical_data['volume_change'] = historical_data['Volume'].diff()

# Calculate balance volume
historical_data['obv'] = (historical_data['Close'] - historical_data['Open'] > 0).astype(int) * historical_data['Volume']
# Calculate cumulative balance volume
historical_data['obv'] = historical_data['obv'].cumsum()

# Calculate transaction amount
historical_data['trade_amount'] = historical_data['Close'] * historical_data['Volume']
# Calculate volume weighted average price
historical_data['vwap'] = historical_data['trade_amount'].cumsum() / historical_data['Volume'].cumsum()


# In[88]:


# plot volume change, balance volume, volume weighted average price
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10,10))
historical_data.plot(y='volume_change', ax=axes[0])
historical_data.plot(y='obv', ax=axes[1])
historical_data.plot(y='vwap', ax=axes[2])
plt.show()


# ### Financial statement indicators

# - Earnings per share (EPS): The ratio of a company's net profit in a certain period to the number of all common shares issued during that period, used to measure the company's profitability. The higher the EPS, the stronger the company's profitability, and investors are more willing to hold the company's stock.
# 
# - Price-to-earnings ratio (P/E): The ratio of a stock's market price to earnings per share (EPS). The price-to-earnings ratio can reflect the market's expectations of a stock's investment value and future profitability. Stocks with high price-to-earnings ratios mean that investors are more willing to pay a high price for the stock in order to obtain higher returns in the future. Stocks with low price-to-earnings ratios are generally considered to be cheaper investment opportunities
# 
# - Return on equity (ROE): Measures the ratio between a company's profits and equity capital. Companies with high ROE represent higher profitability and better returns on shareholders' equity. This indicator is often used to measure a company's management efficiency and is widely considered to be one of the core indicators for measuring a company's financial performance.

# In[603]:


# Calculate basic financial metrics
revenue = Income_data_transposed['Total Revenue']
gross_profit = Income_data_transposed['Gross Profit']
operating_income = Income_data_transposed['Operating Income']
net_income = Income_data_transposed['Net Income from Continuing Oprating']
total_assets = Balance_data_transposed['Total Assets']
total_liabilities = Balance_data_transposed['Total Debt']
total_equity = Balance_data_transposed["Stockholders'Equity"]
cash_from_operations = Cash_flow_data_transposed['Operating Cash Flow']

# Calculate financial ratios
ebitda_margin = (gross_profit - operating_income) / revenue
net_profit_margin = net_income / revenue
return_on_assets = net_income / total_assets
return_on_equity = net_income / total_equity
debt_to_equity_ratio = total_liabilities / total_equity

# Put data into a DataFrame
basic_financial_metrics_data = {
    'Revenue': revenue,
    'Gross Profit': gross_profit,
    'Operating Income': operating_income,
    'Net Income': net_income,
    'Total Assets': total_assets,
    'Total Liabilities': total_liabilities,
    'Total Equity': total_equity,
    'Cash from Operations': cash_from_operations,
    'EBITDA Margin': ebitda_margin,
    'Net Profit Margin': net_profit_margin,
    'Return on Assets': return_on_assets,
    'Return on Equity': return_on_equity,
    'Debt to Equity Ratio': debt_to_equity_ratio,
    "eps":eps,
    "roe":roe
    
}

basic_financial_metrics_data = pd.DataFrame(basic_financial_metrics_data)

# Print the financial data in a table
print(basic_financial_metrics_data)


# #### Calculate financial features

# In[335]:


# net income ＆ average number of common shares ＆ shareholders' equity
net_income = Income_data_transposed['Net Income from Continuing Oprating']
avg_common_shares_outstanding = Balance_data_transposed['Ordinary Shares Number']
shareholders_equity = Balance_data_transposed["Stockholders'Equity"]
stock_price = historical_data['Close']

import pandas as pd

# Load historical data into a DataFrame
historical_data = pd.read_csv('NvidiaStockPrice.csv', index_col='Date', parse_dates=True)
# Replace missing values with previous values
historical_data['Close'] = historical_data['Close'].interpolate(method='time', limit_direction='backward')

# Calculate EPS
eps = net_income / avg_common_shares_outstanding

# Calculate P/E ratio
pe_ratio = stock_price / eps

# Calculate ROE 
roe = net_income / shareholders_equity


# In[324]:


eps


# In[325]:


pe_ratio


# The price-to-earnings ratio calculation failed, so give up this indicator

# In[327]:


roe


# ### News source indicators

# - Sentiment Analysis: Refers to the process of classifying, extracting, and analyzing sentiment in text data based on natural language processing (NLP) technology. In the financial field, sentiment analysis can be used to analyze the attitudes and emotions of market participants towards the market environment, company financial status, and other relevant events. This helps investors better understand market sentiment and make more informed investment decisions.
# 
# - News Volume: Refers to the total number of news reports related to a company, market, or industry published during a specific period. An increase in news volume may indicate that market participants have more interest and concern about major changes in the company or market. This may lead to stock price fluctuations or more trading activities.
# 
# - Event Impact: Refers to the reaction of market participants to specific events or news reports in a company, market, or industry. These events may include financial statement releases, government policy changes, strategic agreements with competitors, large-scale layoffs, etc. Event impact can be measured by analyzing changes in stock prices or other related market activities. This can help investors determine the actual impact of events on companies or markets.

# In[338]:


print(nvidia_sentimental_analisis['sentiment_nn'].value_counts())


# ## Deep Learning Models

# ### Preparing for modeling with deep learning

# In[343]:


print(historical_data.columns)


# In[345]:


historical_data


# In[382]:


historical_data['Date'] = pd.to_datetime(historical_data['Date'])

# Extract the month and year from the 'Date' column
historical_data['Month'] = historical_data['Date'].dt.month
historical_data['Year'] = historical_data['Date'].dt.year


# Calculate the mean close price per month
monthly_mean = historical_data.groupby([ 'Month'])['Close'].mean().reset_index()

# Calculate the mean close price per year
yearly_mean = historical_data.groupby([ 'Year'])['Close'].mean().reset_index()


# In[383]:


monthly_mean


# In[387]:


# Plot the mean close price per month using a Seaborn bar chart
ax =sns.barplot(data=monthly_mean, x='Month', y='Close',  palette='Set3')

plt.title('Mean Close Price per Month')
plt.xlabel('Month')
plt.ylabel('Mean Close Price')

plt.xticks(range(0, 12), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.legend(title='Year')

for bars in ax.containers:
     ax.bar_label(bars, fontsize=10, fmt='%.2f')
plt.show()


# The Mean Close Price per Month in the first eight months is generally higher than that in the last four months.

# In[388]:


yearly_mean.head()


# In[390]:


lst_year= yearly_mean['Year'].tolist()

# Plot the mean close price per month using a Seaborn bar chart
ax = sns.barplot(data=yearly_mean, x='Year', y='Close',  palette='Set3')

plt.title('Mean Close Price per Year')
plt.xlabel('Year')
plt.ylabel('Mean Close Price')

plt.xticks(range(0, len(lst_year)), labels=lst_year)
plt.legend(title='Year')

for bars in ax.containers:
     ax.bar_label(bars, fontsize=10, fmt='%.2f')

plt.show()


# Nvidia's stock price remained at a low level from 2000 to 2011, and grew rapidly after 2012. Its stock price in 2024 has doubled by 2.5 times compared to that in 2023.We selected the stock prices of 2003 and 2024 for analysis

# In[411]:


# Plot the mean close price per month using a Seaborn bar chart
plt.figure(figsize=(12, 6))

filtered_df = historical_data[historical_data['Year'].isin([2003,2024])]
sns.lineplot(data=filtered_df, x='Month', y='Close', hue='Year', palette='Set3')

plt.title('Mean Close Price per Month')

plt.xlabel('Month')
plt.ylabel('Mean Close Price')

plt.xticks(range(1, 13), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

plt.legend(title='Year')
plt.show()


# The bar chart showing the mean close price per years (2018,2023), with each year represented by a different color.
# 
# With our target being Closing price (USD), it is reasonable to assume that there may be some seasonality in our target.
# 
# We know that capture information from the previous seasonal cycle can help us forecast our time series. Therefore, it is important to determine ways to identify seasonality in time series. Usually, plotting the time series data is enough to observe periodic patterns.
# 
# We can plot our target to see if we can visually detect any seasonality.
# 
# Yearly Seasonality
# 
# yearly cyclic behavior refers to a repeating pattern or trend that occurs within each year, the absence of it indicates that there is no noticeable pattern or trend that repeats on an annual basis. This suggests that the stock market data does not exhibit consistent fluctuations or variations that occur predictably from one year to the next. There may be other factors or irregularities at play that overshadow any yearly cyclic behavior.

# In[416]:


fig, ax = plt.subplots()

ax.plot(historical_data['Close'])
for i in np.arange(0, len(historical_data), 260):
    ax.axvline(x=i, linestyle='--', color='black', linewidth=1)

plt.title('Yearly Seasonality')
plt.xlabel('Year')
plt.ylabel('Seasonal Component')

plt.xticks(np.arange(67, len(historical_data), 260), np.arange(2000, 2024, 1))

plt.figure(figsize=(10, 5))

fig.autofmt_xdate()
plt.tight_layout()


# Highlighting the seasonal pattern in the Yearly Close Price. The dashed vertical lines separate periods of 260 working days per year.
# 
# We can clearly see the yearly frequency does not show any cyclic behavior. Therefore, there is no yearly seasonality
# 
# Another way of identifying seasonal patterns in a time series is using time series decomposition,We can decompose the dataset for Stock Prices using the tsa.seasonal_decompose function from the statsmodels library
# 
# Monthly Seasonality
# 
# Monthly cyclic behavior refers to a repeating pattern or trend that occurs within each month. It suggests that there are consistent fluctuations or variations in the stock market data that repeat on a monthly basis. This could be due to various factors, such as economic events, market sentiment, or trading patterns that occur within specific months.

# In[419]:


df_seasonality = historical_data.copy()

# Set 'Date' column as the index of the DataFrame
df_seasonality.set_index('Date', inplace=True)

# Resample the data to monthly frequency
monthly_data = df_seasonality['Close'].resample('M').mean()

# Get the month numbers for the xticks
month_numbers = pd.to_datetime(monthly_data.index).strftime('%m')

df_monthly_data= monthly_data.to_frame(name="Close")
df_monthly_data['month_numbers'] = month_numbers

# Perform seasonal decomposition
decomposition = sm.tsa.seasonal_decompose(monthly_data, model='additive')

# Plot the seasonal component
fig, ax = plt.subplots()

decomposition.seasonal.plot(ax=ax)

times = np.arange('2000-01', '2024-08', dtype='datetime64[M]')
ax.set_xticks(times)
# ax.set_xticklabels(times )

for i in np.arange(0, len(historical_data), 1):
    ax.axvline(x=i, linestyle='--', color='black', linewidth=1)

plt.title('Monthly Seasonality' ,  fontsize = 12)
plt.xlabel('Month',  fontsize = 10)
plt.ylabel('Seasonal Component',fontsize = 10)

plt.rcParams['figure.figsize'] = (30, 10)

fig.autofmt_xdate()
ax.set_xticklabels(times,rotation = 75,
                   ha = 'right', fontsize = 12,
                   color = 'blue')

plt.tight_layout()
plt.show()


# We can see that in our stock market data, recurring patterns indicate monthly seasonality. The dashed vertical lines separate the one-month periods for each year.
# 
# By looking at the total closing prices for each month, it is easy to identify a recurring pattern for each month, with February and June closing significantly higher each year, while January and September close lower each year.
# 
# This observation is often enough to determine that a data set is seasonal and indicates that there is a monthly pattern or seasonality in the data.
# 
# According to tradethatswing, these are the best and worst months for the stock market - seasonal patterns

# In[426]:


# 'Month','Year',
cols_to_drop = ['Date']
historical_data = historical_data.drop(cols_to_drop, axis=1)

historical_data.head()


# In[427]:


# Splitting and scaling the data 
n = len(historical_data)

# Split 70:20:10 (train:validation:test)
train_df = historical_data[0:int(n*0.7)]
val_df = historical_data[int(n*0.7):int(n*0.9)]
test_df = historical_data[int(n*0.9):]


# In[428]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(train_df)

train_df[train_df.columns] = scaler.transform(train_df[train_df.columns])
val_df[val_df.columns] = scaler.transform(val_df[val_df.columns])
test_df[test_df.columns] = scaler.transform(test_df[test_df.columns])


# In[429]:


train_df.to_csv('../train.csv', index=False, header=True)
val_df.to_csv('../val.csv', index=False, header=True)
test_df.to_csv('../test.csv', index=False, header=True)


# In[431]:


column_indices = {name: i for i, name in enumerate(train_df.columns)}


# ### No Sentiment Data Denoising Modeling with Deep Learning

# DataWindow class allows us to quickly create windows of data for training deep learning models. Each window of data contains a set of inputs and a set of labels. The model is then trained to produce predictions as close as possible to the labels using the inputs.time windows

# In[432]:


class DataWindow():

  def __init__(self, input_width, label_width, shift,
                 train_df=train_df, val_df=val_df, test_df=test_df,
                 label_columns=None):

            self.train_df = train_df
            self.val_df = val_df
            self.test_df = test_df

            #Name of the column that we wish to predict
            self.label_columns = label_columns
            if label_columns is not None:
                #Create a dictionary with the name and index of the label column. This will be used for plotting.
                self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
            #Create a dictionary with the name and index of each column. This will be used to separate the features from the target variable
            self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

            self.input_width = input_width
            self.label_width = label_width
            self.shift = shift

            self.total_window_size = input_width + shift

            #The slice function returns a slice object that specifies how to slice a sequence.
            # In this case, it says that the input slice starts at 0 and ends when we reach the input_width.
            self.input_slice = slice(0, input_width)
            #Assign indices to the inputs. These are useful for plotting.
            self.input_indices = np.arange(self.total_window_size)[self.input_slice]


            #Get the index at which the label starts. In this case, it is the total window size minus the width of the label.
            self.label_start = self.total_window_size - self.label_width
            #The same steps that were applied for the inputs are applied for labels.
            self.labels_slice = slice(self.label_start, None)
            self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

  def split_to_inputs_labels(self, features):
            #Slice the window to get the inputs using the input_slice defined in __init__.
            inputs = features[:, self.input_slice, :]
            #Slice the window to get the labels using the labels_slice defined in __init__
            labels = features[:, self.labels_slice, :]

            #If we have more than one target, we stack the labels.
            if self.label_columns is not None:
                labels = tf.stack(
                    [labels[:,:,self.column_indices[name]] for name in self.label_columns],
                    axis=-1
                )
            #The shape will be [batch, time, features].
            # At this point,we only specify the time dimension and allow the batch and feature dimensions to be defined later.
            inputs.set_shape([None, self.input_width, None])
            labels.set_shape([None, self.label_width, None])

            return inputs, labels

  def plot(self, model=None, plot_col='Close', max_subplots=3):
            inputs, labels = self.sample_batch

            plt.figure(figsize=(12, 8))
            plot_col_index = self.column_indices[plot_col]
            max_n = min(max_subplots, len(inputs))

            #Plot the inputs. They will  appear as a continuous blue line with dots.
            for n in range(max_n):
                plt.subplot(3, 1, n+1)
                plt.ylabel(f'{plot_col} [scaled]')
                plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                        label='Inputs', marker='.', zorder=-10)

                if self.label_columns:
                  label_col_index = self.label_columns_indices.get(plot_col, None)
                else:
                  label_col_index = plot_col_index

                if label_col_index is None:
                  continue

                #Plot the labels or actual. They will appear as green squares.
                plt.scatter(self.label_indices, labels[n, :, label_col_index],
                            edgecolors='k', marker='s', label='Labels', c='green', s=64)
                if model is not None:
                  predictions = model(inputs)
                  #Plot the predictions. They will appear as red crosses.
                  plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                              marker='X', edgecolors='k', label='Predictions',
                              c='red', s=64)

                if n == 0:
                  plt.legend()

            plt.xlabel('Date (Day)')
            plt.ylabel('Closing price (USD)')

  def make_dataset(self, data):
            data = np.array(data, dtype=np.float32)
            ds = tf.keras.preprocessing.timeseries_dataset_from_array(
                #Pass in the data. This corresponds to our training set, validation set, or test set.
                data=data,
                #Targets are set to None, as they are handled by the split_to_input_labels function.
                targets=None,
                #Define the total length of s the array, which is equal to the total window length.
                sequence_length=self.total_window_size,
                #Define the number of timesteps separating each sequence. In our case, we want the sequences to be consecutive, so sequence_stride=1.
                sequence_stride=1,
                #Shuffle the sequences. Keep in mind that the data is still in chronological order. We are simply shuffling the order of the sequences, which makes the model more robus
                shuffle=True,
                #Define the number of sequences in a single batch
                batch_size=32
            )

            ds = ds.map(self.split_to_inputs_labels)
            return ds

  @property
  def train(self):
      return self.make_dataset(self.train_df)

  @property
  def val(self):
      return self.make_dataset(self.val_df)

  @property
  def test(self):
      return self.make_dataset(self.test_df)

  @property
  def sample_batch(self):
    #Get a sample batch of data for plotting purposes. If the sample batch does not exist, we’ll retrieve a sample batch and cache it
      result = getattr(self, '_sample_batch', None)
      if result is None:
          result = next(iter(self.train))
          self._sample_batch = result
      return result


# ### Utility function to train models

# In[434]:


#The function takes a model, and a window of data from the DataWindow class.
# The patience: is the number of epochs after which the model should stop training if the validation loss does not improve;
# max_epochs: sets a maximum number of epochs to train the model.
def compile_and_fit(model, window, patience=3, max_epochs=50):

    #Early stopping occurs if 3 consecutive epochs do not decrease the validation loss, as set by the patience parameter
    # The validation loss is tracked to determine if we should apply early stopping or not.
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=patience,
                                   mode='min')

    # The MSE is used as the loss function.
    model.compile(loss=MeanSquaredError(),
                  optimizer=Adam(),
                  metrics=[MeanAbsoluteError()]) #the MAE as an evaluation metric to compare the performance of our models

    # The model is fit on the training set.
    history = model.fit(window.train,
                       epochs=max_epochs,   #The model can train for at most 50 epochs, as set by the max_epochs parameter.
                       validation_data=window.val,
                       callbacks=[early_stopping]) #early_stopping is passed as a callback. If the validation loss does not decrease after 3 consecutive epochs, the model stops training. This avoids overfitting.

    return history


# ### Modeling with Deep Learning

# We will build:
# 
# - Two baselines,
# - A linear model,
# - A deep neural network model,
# - A longshort-term memory (LSTM) model,
# - A convolutional neural network (CNN),
# - A combination of CNN and LSTM, and
# - An autoregressive LSTM.
# 
# In the end, we will use the mean absolute error (MAE) to determine which model is the best. The one that achieves the lowest MAE on the test set will be the top-performing model.
# Let's read the training set, validation set, and test set so they are ready for modeling

# #### First: Baseline models 

# Every forecasting project must start with a baseline model. Baselines serve as a benchmark for our more sophisticated models, as they can only be better in comparison to a certain benchmark.we’ll build two baseline models:
# 
# - a. One that repeats the last known value (last day) and,
# - b. Another that repeats the last month (21 days) of data.
# 
# We’ll start by creating the window of data that will be used. Recall that the objective is to forecast the next 21 working days of Close price, Thus:
# 
# - the length of our `label sequence` is 21 timesteps, 
# - the `shift` will also be 21 timesteps, and
# - We’ll also use an `input length` of 21.

# In[435]:


multi_window = DataWindow(input_width=21, label_width=21, shift=21,label_columns=['Close'])


# ####  a.Repeat last value - Baseline

# To predict the last known value, We implement MultiStepLastBaseline class that simply takes in the input and repeats the last value of the input sequence over 21 timesteps. This acts as the prediction of the model.

# In[468]:


class MultiStepLastBaseline(Model):
    def __init__(self, label_index=None):
        super().__init__()
        self.label_index = label_index

    def call(self, inputs):
        if self.label_index is None:
            return tf.tile(inputs[:, -1:, :], [1, 21, 1])
        return tf.tile(inputs[:, -1:, self.label_index:], [1, 21, 1])


# In[469]:


baseline_last = MultiStepLastBaseline(label_index=column_indices['Close'])

baseline_last.compile(loss=MeanSquaredError(), metrics=[MeanAbsoluteError()])

val_performance = {}
performance = {}

val_performance['Baseline - Last'] = baseline_last.evaluate(multi_window.val)
performance['Baseline - Last'] = baseline_last.evaluate(multi_window.test, verbose=0)


# In[470]:


multi_window.plot(baseline_last)


# Predictions from the baseline model, which simply repeats the last known input value (last day) , The inputs are shown with square markers, and the labels are shown with crosses. Each data window consists of 21 timesteps with square markers followed by 21 labels with crosses.

# #### b. Repeat last day - Baseline

# Next, let’s implement a baseline model that repeats the input sequence (21 day),This means that the prediction for the next 21 days will simply be the last known 21 days of data as it is.

# In[500]:


class RepeatBaseline(Model):
    def __init__(self, label_index=None):
        super().__init__()
        self.label_index = label_index

    def call(self, inputs):
        return inputs[:, :, self.label_index:]


# In[501]:


baseline_repeat = RepeatBaseline(label_index=column_indices['Close'])

baseline_repeat.compile(loss=MeanSquaredError(), metrics=[MeanAbsoluteError()])

val_performance['Baseline - Repeat'] = baseline_repeat.evaluate(multi_window.val)
performance['Baseline - Repeat'] = baseline_repeat.evaluate(multi_window.test, verbose=0)


# In[502]:


multi_window.plot(baseline_repeat)


# We can see that the predictions are equal to the input sequence, which is the expected behavior for this baseline model.

# #### Second: Linear model 

# In[474]:


label_index = column_indices['Close']
num_features = train_df.shape[1]

linear = Sequential([
    Dense(1, kernel_initializer=tf.initializers.zeros)
])


# In[475]:


history = compile_and_fit(linear, multi_window)

val_performance['Linear'] = linear.evaluate(multi_window.val)
performance['Linear'] = linear.evaluate(multi_window.test, verbose=0)


# In[476]:


multi_window.plot(linear)


# Predictions generated from a linear model

# #### hird: Deep neural network - Dense Model

# In[477]:


dense = Sequential([
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, kernel_initializer=tf.initializers.zeros),
])

history = compile_and_fit(dense, multi_window)

val_performance['Dense'] = dense.evaluate(multi_window.val)
performance['Dense'] = dense.evaluate(multi_window.test, verbose=0)


# In[478]:


multi_window.plot(dense)


# Predections genertaed from Dense Model

# #### Fourth: Long short-term memory (LSTM) model 

# Long short-term memory (LSTM) is a deep learning architecture that is a subtype of RNN. LSTM addresses the problem of short-term memory by adding the cell state. This allows for past information to flow through the network for a longer period of time, meaning that the network still carries information from early values in the sequence.
# 
# The main advantage of the long short-term memory (LSTM) model is that it keeps information from the past in memory. This makes it especially suitable for treating sequences of data, like time series. It allows us to combine information from the present and the past to produce a prediction.
# 
# - We’ll feed the input sequence through an LSTM layer before sending it to the output layer, which remains a Dense layer with one neuron.
# - We’ll then train the model and store its performance in the dictionary for comparison at the end.

# In[479]:


lstm_model = Sequential([
    LSTM(32, return_sequences=True),
    Dense(1, kernel_initializer=tf.initializers.zeros),
])

history = compile_and_fit(lstm_model, multi_window)

val_performance['LSTM'] = lstm_model.evaluate(multi_window.val)
performance['LSTM'] = lstm_model.evaluate(multi_window.test, verbose=0)


# In[480]:


multi_window.plot(lstm_model)


# Predictions generated from the LSTM model.

# #### Fifth: Convolutional neural network (CNN)

# Convolutional neural network (CNN)
# 
# - A convolutional neural network (CNN) is a deep learning architecture that uses the convolution operation. This allows the network to reduce the feature space, effectively filtering the inputs and preventing overfitting.
# - The convolution is performed with a kernel, which is also trained during model fitting. The stride of the kernel determines the number of steps it shifts at each step of the convolution. In time series forecasting, only 1D convolution is used.
# - To avoid reducing the feature space too quickly, we can use padding, which adds zeros before and after the input vector. This keeps the output dimension the same as the original feature vector, allowing us to stack more convolution layers, which in turn allows the network to process the features for a longer time. AThe main advantage of convolutional neural network (CNN) is that it uses the convolution function to reduce the feature space. This effectively filters our time series and performs feature selection. Furthermore, a CNN is faster to train than an LSTM since the operations are parallelized, whereas the LSTM must treat one element of the sequence at a time.
# 
# Because the convolution operation reduces the feature space, we must provide a slightly longer input sequence to make sure that the output sequence contains 21 timesteps. How much longer it needs to be depends on the length of the kernel that performs the convolution operation. In this case, we’ll use a kernel length of 3.
# 
# This is an arbitrary choice, Given that we need 21 labels, we can calculate the input sequence using equation:
# 
# $$input length = label length + kernel length – 1$$
# 
# This forces us to define a window of data specifically for the CNN model. Note that since we are defining a new window of data, the sample batch used for plotting will differ from the one used so far.

# In[481]:


KERNEL_WIDTH = 3
LABEL_WIDTH = 21
INPUT_WIDTH = LABEL_WIDTH + KERNEL_WIDTH - 1


# In[482]:


cnn_multi_window = DataWindow(input_width=INPUT_WIDTH, label_width=LABEL_WIDTH, shift=21, label_columns=['Close'])


# In[483]:


cnn_model = Sequential([
    Conv1D(32, activation='relu', kernel_size=(KERNEL_WIDTH)),
    Dense(units=32, activation='relu'),
    Dense(1, kernel_initializer=tf.initializers.zeros),
])

history = compile_and_fit(cnn_model, cnn_multi_window)

val_performance['CNN'] = cnn_model.evaluate(cnn_multi_window.val)
performance['CNN'] = cnn_model.evaluate(cnn_multi_window.test, verbose=0)


# In[484]:


cnn_multi_window.plot(cnn_model)


# shows that the input sequence differs from our previous methods because working with a CNN involves windowing the data again to account for the convolution kernel length. The training, validation, and test sets remain unchanged, so it is still valid to compare all the models’ performance.

# #### Sixth: Combining a CNN with an LSTM

# We know that:
# 
# - LSTM is good at treating sequences of data,
# - While CNN can filter a sequence of data.
# Therefore, it is a reasonable hypothesis that filtering our input sequence before feeding it to an LSTM might improve the performance, It is interesting to test whether combining CNN with LSTM can result in a better-performing model.
# 
# We’ll:
# 
# - Feed the input sequence to a Conv1D layer,but
# - Use an LSTM layer for learning this time.
#   
# Then we’ll send the information to the output layer. Again, we’ll train the model and store its performance.

# In[485]:


cnn_lstm_model = Sequential([
    Conv1D(32, activation='relu', kernel_size=(KERNEL_WIDTH)),
    LSTM(32, return_sequences=True),
    Dense(1, kernel_initializer=tf.initializers.zeros),
])


# In[486]:


history = compile_and_fit(cnn_lstm_model, cnn_multi_window)

val_performance['CNN + LSTM'] = cnn_lstm_model.evaluate(cnn_multi_window.val)
performance['CNN + LSTM'] = cnn_lstm_model.evaluate(cnn_multi_window.test, verbose=0)


# In[487]:


cnn_multi_window.plot(cnn_lstm_model)


# Predictions from a CNN combined with an LSTM model

# #### Seventh: The Autoregressive LSTM model - Using Predictions To Make More Predictions

# The final model that we’ll implement is an autoregressive LSTM (ARLSTM) model. Instead of generating the entire output sequence in a single shot, the Autoregressive model will generate one prediction at a time and use that prediction as an input to generate the next one.
# 
# This kind of architecture is present in state-of-the-art forecasting models, but it is worth nothing that Autoregressive deep learning models come with a major caveat, which is the accumulation of error. If the model generates a very bad first prediction, this mistake will be carried on to the next predictions, which will magnify the errors. That error accumulates as it is fed back into the model, meaning that later predictions will have a larger error than earlier predictions. Nevertheless, it is worth testing this model and have this model in our toolbox of time series forecasting methods.
# 
# Let's see if it works well in our situation.
# 
# The first step is defining the class that implements the ARLSTM model.

# In[488]:


class AutoRegressive(Model):
    def __init__(self, units, out_steps):
        super().__init__()
        self.out_steps = out_steps
        self.units = units
        self.lstm_cell = LSTMCell(units)
        self.lstm_rnn = RNN(self.lstm_cell, return_state=True)
        self.dense = Dense(train_df.shape[1])

    def warmup(self, inputs):
        x, *state = self.lstm_rnn(inputs)
        prediction = self.dense(x)

        return prediction, state

    def call(self, inputs, training=None):
        predictions = []
        prediction, state = self.warmup(inputs)

        predictions.append(prediction)

        for n in range(1, self.out_steps):
            x = prediction
            x, state = self.lstm_cell(x, states=state, training=training)

            prediction = self.dense(x)
            predictions.append(prediction)

        predictions = tf.stack(predictions)
        predictions = tf.transpose(predictions, [1, 0, 2])

        return predictions


# In[489]:


AR_LSTM = AutoRegressive(units=32, out_steps=21)


# In[490]:


history = compile_and_fit(AR_LSTM, multi_window)

val_performance['ARLSTM'] = AR_LSTM.evaluate(multi_window.val)
performance['ARLSTM'] = AR_LSTM.evaluate(multi_window.test, verbose=0)


# In[491]:


multi_window.plot(AR_LSTM)


# Predictions from the ARLSTM model

# #### Selecting Best Model

# In[503]:


mae_val = [v[1] for v in val_performance.values()]
mae_test = [v[1] for v in performance.values()]


# In[504]:


x = np.arange(len(performance))

fig, ax = plt.subplots()
ax.bar(x - 0.15, mae_val, width=0.25, color='black', edgecolor='black', label='Validation')
ax.bar(x + 0.15, mae_test, width=0.25, color='white', edgecolor='black', hatch='/', label='Test')
ax.set_ylabel('Mean absolute error')
ax.set_xlabel('Models')


font_prop = font_manager.FontProperties( size=20)

for index, value in enumerate(mae_val):
    plt.text(x=index - 0.15, y=value+0.005, s=str(round(value, 3)), ha='center',fontproperties=font_prop)

for index, value in enumerate(mae_test):
    plt.text(x=index + 0.15, y=value+0.0025, s=str(round(value, 3)), ha='center',fontproperties=font_prop)

# plt.ylim(0, 0.33)
plt.xticks(ticks=x, labels=performance.keys())
plt.legend(loc='best')

plt.legend(fontsize=25) # using a size in points

plt.tight_layout()
plt.figure(figsize=(10,6))


#  Comparing the MAE of all models tested

# In[505]:


model_names = ['Baseline-Last', 'Baseline-Repeat', 'Linear', 'Dense', 'LSTM', 'CNN', 'LSTM+CNN', 'ARLSTM']
data = {'Test - MAE': mae_test, 'Validation - MAE': mae_val}
df = pd.DataFrame(data, index=model_names)
df_sorted = df.sort_values(by='Test - MAE', ascending=True)
df_sorted.T


# Undenoised dataset: On the test set, we can see that Baseline-RepeatLSTM and LinearBaseline-LastARLSTMDense have lower MAE values, while CNNLSTM and CNN have higher MAE values. This indicates that Baseline-RepeatLSTM and LinearBaseline-LastARLSTMDense are more suitable for this dataset, and CNNLSTM and CNN may need more optimization to work better on this dataset again.
# 
# On the validation set, we can see that the MAE values ​​of all schemes are in a similar range except for the last data point. This indicates that the performance differences of these schemes are not very obvious in the validation set. However, on the last data point, the CNN scheme has a higher MAE value and may need more tuning to adapt to this dataset.

# ### Sentiment analysis after noise reduction test

# In[520]:


# Load datasets
sentiment_data = pd.read_csv('nvidia_sentimental_analisis.csv')
stock_price_data = pd.read_csv('NvidiaStockPrice.csv')

sentiment_data = sentiment_data.replace({'sentiment_rf': {'neutral': 0, 'positive': 1, 'negative': -1}})
sentiment_data = sentiment_data.replace({'sentiment_nn': {'neutral': 0, 'positive': 1, 'negative': -1}})
sentiment_data['sentiment_rf'] = sentiment_data['sentiment_rf'].astype('int')
sentiment_data['sentiment_nn'] = sentiment_data['sentiment_nn'].astype('int')

# Quick data exploration
print(sentiment_data.head())
print(stock_price_data.head())

# Check for missing values
print(sentiment_data.isnull().sum())
print(stock_price_data.isnull().sum())


# In[521]:


# Convert 'Date' to datetime format
sentiment_data['Date'] = pd.to_datetime(sentiment_data['Date'])
stock_price_data['Date'] = pd.to_datetime(stock_price_data['Date'])

# Merge datasets
merged_data = pd.merge(stock_price_data, sentiment_data, on='Date', how='left')


# In[522]:


# Apply moving average to smooth stock price
merged_data['Close_SMA_10'] = merged_data['Close'].rolling(window=10).mean()

# Fill missing sentiment scores with interpolation or rolling means
merged_data['sentiment_rf'].fillna(merged_data['sentiment_rf'].rolling(window=5).mean(), inplace=True)
merged_data['sentiment_nn'].fillna(merged_data['sentiment_nn'].rolling(window=5).mean(), inplace=True)


# In[526]:


# Lag features
merged_data['Close_Lag_1'] = merged_data['Close'].shift(1)
merged_data['Close_Lag_2'] = merged_data['Close'].shift(2)

# Rolling window features
merged_data['sentiment_rf_score_Rolling_7'] = merged_data['sentiment_rf'].rolling(window=7).mean()
merged_data['sentiment_nn_score_Rolling_7'] = merged_data['sentiment_nn'].rolling(window=7).mean()


# In[527]:


# Define the target variable
merged_data['Target'] = merged_data['Close'].shift(-1)
merged_data.dropna(inplace=True)  # Drop rows with NaN values after shifting


# In[533]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Selecting relevant features
features = ['Close', 'Close_SMA_10', 'sentiment_rf', 'sentiment_nn', 'Close_Lag_1', 'Close_Lag_2']
target = 'Target'

# Scaling data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(merged_data[features + [target]])

# Splitting data into training and testing sets
train_size = int(len(scaled_data) * 0.8)
train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

# Reshape data for LSTM model
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, -1])
    return np.array(X), np.array(y)

seq_length = 10
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)


# In[534]:


# LSTM Model Implementation
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# Define LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, shuffle=False)


# In[535]:


# Hybrid Model Implementation
from keras.layers import Conv1D, MaxPooling1D, Flatten

# Define Hybrid CNN-LSTM model
hybrid_model = Sequential()
hybrid_model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
hybrid_model.add(MaxPooling1D(pool_size=2))
hybrid_model.add(LSTM(50, return_sequences=False))
hybrid_model.add(Dropout(0.2))
hybrid_model.add(Dense(1))

# Compile the hybrid model
hybrid_model.compile(optimizer='adam', loss='mean_squared_error')

# Train the hybrid model
history_hybrid = hybrid_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, shuffle=False)


# In[546]:


#Model Evaluation
# Make predictions
predictions_lstm = model.predict(X_test)
predictions_hybrid = hybrid_model.predict(X_test)

# Rescale predictions back to original values
predictions_lstm_rescaled = scaler.inverse_transform(np.concatenate([X_test[:, -1, :-1], predictions_lstm], axis=1))[:, -1]
predictions_hybrid_rescaled = scaler.inverse_transform(np.concatenate([X_test[:, -1, :-1], predictions_hybrid], axis=1))[:, -1]

# Calculate performance metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("LSTM Model:")
print(f"MSE: {mean_squared_error(y_test, predictions_lstm)}")
print(f"MAE: {mean_absolute_error(y_test, predictions_lstm)}")
print(f"R2: {r2_score(y_test, predictions_lstm)}")

print("Hybrid Model (CNN-LSTM):")
print(f"MSE: {mean_squared_error(y_test, predictions_hybrid)}")
print(f"MAE: {mean_absolute_error(y_test, predictions_hybrid)}")
print(f"R2: {r2_score(y_test, predictions_hybrid)}")


# First, MSE is an indicator that evaluates the average value of the square of the difference between the model prediction value and the true value. The smaller its value, the smaller the model prediction error. Here, it can be observed that the MSE value of the Hybrid Model is smaller than that of the LSTM Model, indicating that it performs better in data prediction.
# 
# Second, MAE is an indicator that measures the average value of the absolute value of the difference between the model prediction value and the true value. The smaller its value, the smaller the model prediction error. Here, the MAE value of the Hybrid Model is also smaller than that of the LSTM Model, indicating that the prediction performance of the Hybrid Model is better.
# 
# Finally, R2 is an indicator that measures the model's ability to explain data variance. Its value is in the range of [0, 1]. The closer the value is to 1, the better the model's ability to explain data. Here, neither model reaches R2 = 1, but the R2 value of the LSTM Model is closer to 1, which means that it is slightly better in explaining data variance.
# 
# Therefore, the MSE and MAE values ​​of the Hybrid Model are smaller than those of the LSTM Model, but the LSTM Model is slightly better than the Hybrid Model in explaining data variance. When selecting a model, you need to balance the requirements of different indicators according to the actual situation and comprehensively consider the selection of the most appropriate model.

# ### Asset bubbles detecting

# In[634]:


# Based on the comparison between the current stock price and the average price of the past 5-10 years and the current 3-year Treasury bond yield


# In[635]:


import pandas as pd

# Read historical stock price data
df = pd.read_csv('NvidiaStockPrice.csv')

# Calculate the average price of stock prices in the past 5-10 years
mean_price_past_5_years = df['Close'].loc['2015-01-01':'2020-12-31'].mean()
mean_price_past_10_years = df['Close'].loc['2010-01-01':'2020-12-31'].mean()

# Get the current stock price
current_price = df['Close'][1]

# Read 3-year Treasury bond yield data
rate_df = pd.read_csv('3m_treasury_bill_interest_rates.csv')

# Get the latest 3-year Treasury bond yield
latest_rate = rate_df['Close'][1]

# Based on the comparison between the current stock price and the average price of the past 5-10 years and the current 3-year Treasury bond yield, determine whether there is a stock price bubble
if current_price >= mean_price_past_5_years and current_price >= mean_price_past_10_years and latest_rate <= 0.03:
    print("There may be a stock price bubble at present")
else:
    print("The current stock price is relatively reasonable")


# ## Results and Discussions

# After reducing the noise of financial news, the Hybrid Model (CNN-LSTM) is more suitable for predicting stock prices. By comparing with the US 3-year Treasury bond yield, Nvidia's stock price is currently reasonable.
