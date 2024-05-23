from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
# import sklearn.preprocessing


start = '2010-01-01'
end = '2020-12-31'

st.title('Stock Price Prediction')

user_input = st.text_input('Enter Stock ticker', 'AAPL.US')
df = data.DataReader(user_input, 'stooq', start, end)

# Describing data
st.subheader('Data from 2010 - 2019')
st.write(df.describe())

# Visualization
st.subheader('Closing Price Vs Time Chart')
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Closing Price Vs Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price Vs Time Chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df.Close)
st.pyplot(fig)

# Spliting the data into training and testing
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])


scaler = MinMaxScaler(feature_range=(0, 1))
data_training_array = scaler.fit_transform(data_training)


# x_train = []  # first 100 days
# y_train = []  # Next 101th day prediction

# for i in range(100, data_training_array.shape[0]):
#     x_train.append(data_training_array[i-100: i])
#y_train.append(data_training_array[i, 0])


#x_train, y_train = np.array(x_train), np.array(y_train)


# load My model
model = load_model('keras_model.h5')

# testing part
past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []  # first 100 days
y_test = []  # Next 101 day prediction

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicated = model.predict(x_test)
scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicated = y_predicated * scale_factor
y_test = y_test * scale_factor

# Final Graph
st.subheader('Predictions Vs Original')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label='original Price')
plt.plot(y_predicated, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.xlabel('Price')
plt.legend()
st.pyplot(fig2)
