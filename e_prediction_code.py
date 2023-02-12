#!/usr/bin/env python
# coding: utf-8

# In[83]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Create the time series data
sales = np.array([1, 1,1,3,4,5,6,7,6,5,4,3,5,8,6,4,2,5,2,1,1,1,2,2,2,1,1,2,1,1,1,1,1,1,0])

sales


# In[84]:


# Create a dataframe to store the time series data
df = pd.DataFrame({'sales': sales})


# In[85]:


# Create a date range for all months of 2020 to 2022
date_rng = pd.date_range(start='2020-01-01', end='2022-12-01', freq='M')
df['date'] = date_rng


# In[86]:


# Set the date column as the index
df.set_index('date', inplace=True)

# Split the data into train and validation sets
train = df[:'2022-01-01']
validate = df['2022-01-01':]


# In[87]:


train


# In[88]:


validate


# In[89]:


# Train the linear regression model
model = LinearRegression()
model.fit(np.array(train.index.astype(int).values.reshape(-1, 1)/10**9).astype(int), train['sales'])


# In[90]:


# Generate the prediction for all months of 2023
date_rng_2023 = pd.date_range(start='2023-01-01', end='2023-07-01', freq='M')
prediction = model.predict(np.array(date_rng_2023.astype(int).values.reshape(-1, 1)/10**9).astype(int))


# In[91]:


prediction


# In[92]:


# Plot the train, validate and prediction data
plt.plot(train.index, train['sales'], label='train')
plt.plot(validate.index, validate['sales'], label='validate')
plt.plot(date_rng_2023, prediction, label='prediction')
plt.legend()
plt.show()


# In[ ]:




