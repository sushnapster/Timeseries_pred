import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Define the time series data
data = [1, 1, 1, 3, 4, 5, 6, 7, 6, 5, 4, 3, 5, 8, 6, 4, 2, 5, 2, 1, 1, 1, 2, 2, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1,1,0]

# Convert the data into a pandas dataframe
df = pd.DataFrame(data, columns=['sales'])

# Add a time index to the dataframe
df['time'] = pd.date_range(start='2020-01-01', end='2022-12-31', freq='M')

# Set the time column as the index
df.set_index('time', inplace=True)

# Split the data into training and validation sets
train = df['2021-01-01':'2021-12-31'].values.astype('datetime64[D]')
validation = df['2022-01-01':'2022-12-31'].values.astype('datetime64[D]')

# Train the linear regression model
model = LinearRegression()
model.fit(train.index.to_frame(), train)
print(model)

## Make predictions on the validation set
#predictions = model.predict(validation.index.to_frame())
#
## Calculate the mean squared error
#mse = mean_squared_error(validation, predictions)
#
## Print the mean squared error
#print("Mean Squared Error:", mse)
#
## Predict the sales revenue for 2023
#x = np.array(['2023-01-01', '2023-12-31'], dtype='datetime64[ns]').reshape(-1, 1)
#y = model.predict(x)
#print("Prediction for 2023:", y)
#
## Plot the train, validation, and prediction data
#plt.plot(train, label='Train')
#plt.plot(validation, label='Validation')
#plt.plot(x, y, label='Prediction')
#plt.legend(loc='best')
#plt.xlabel('Time')
#plt.ylabel('Sales')
#plt.title('Sales Prediction')
#plt.show()