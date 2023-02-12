import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# Create a dataframe from the list of events data
data = [1, 1,1,3,4,5,6,7,6,5,4,3,5,8,6,4,2,5,2,1,1,1,5,6,2,7,5,2,7,8,5,8,9,3,4]
df = pd.DataFrame(data, columns=['events'])

# Divide the data into a training set and a validation set
train_data = df[:24].values
validation_data = df[24:].values

# Reshape the data into a 3D array to be input into the LSTM
train_data = np.reshape(train_data, (train_data.shape[0], 1, train_data.shape[1]))
validation_data = np.reshape(validation_data, (validation_data.shape[0], 1, validation_data.shape[1]))

# Define the LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(train_data.shape[1], train_data.shape[2])))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(train_data, df[:24], epochs=100, batch_size=1, verbose=2)

# Make predictions on the validation set
predictions = model.predict(validation_data)

# Convert the predictions back to a 1D array
predictions = [item for sublist in predictions for item in sublist]

# Print the predictions
print(predictions)

############################################ Future prediction #############################

# Define the number of steps to predict in the future
future_steps = 6

# Get the last `future_steps` values from the end of the training data
last_values = train_data[-future_steps:]

# Reshape the data to be 3-dimensional
last_values = last_values.reshape(1, future_steps, 1)

print('---------------------------------------------')
print(last_values)
# Use the model to make predictions for the next `future_steps` values
future_predictions = model.predict(last_values)

# Reshape the predictions to a 1-dimensional array
future_predictions = np.squeeze(future_predictions)

# Print the predictions
print(future_predictions)

#############################################  Accuracy #####################################

# Convert the predictions and actual events data to numpy arrays
predictions = np.array(predictions)
validation_data = np.array(validation_data)

# Calculate the mean absolute error
mae = np.mean(np.abs(predictions - validation_data))

# Calculate the root mean squared error
rmse = np.sqrt(np.mean((predictions - validation_data)**2))

# Print the mean absolute error and root mean squared error
print("Mean Absolute Error: ", mae)
print("Root Mean Squared Error: ", rmse)



import matplotlib.pyplot as plt

# Plot the actual events data from 2022
plt.plot(np.squeeze(validation_data), label='Actual events')

# Plot the predicted events for 2023
plt.plot(predictions, label='Predicted events')

# Add a legend to the plot
plt.legend()

# Show the plot
plt.show()