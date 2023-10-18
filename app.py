from datetime import datetime,date
import numpy as np
from pandas_datareader import data as pdr
import streamlit as sl
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objs as go

loadedModel = load_model('m.h5')

def get_index_from(target):
  i = 0
  for date, data in df.iterrows():
    if (str(date.date()) == target.strftime('%Y-%m-%d')):
      return (i)
    i += 1
  return (None)

def predict(str):
    data = df.filter(['Close'])

    dataset = data.values
  
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)

    target_index = get_index_from(str)

    actual_price = df['Close'][target_index]

    if (target_index == None):
        return (None)
    x_in = []
    x_in.append(scaled_data[target_index - 10: target_index, :])
    x_in = np.array(x_in)
    predict = loadedModel.predict(x_in)
    predict = scaler.inverse_transform(predict)
    return (predict[0][0], actual_price, abs(float(predict[0][0] - float(actual_price))))


sl.title('AAPL Stock Prediction')

yf.pdr_override()

min_date = date(2000,1,3)
max_date = date(2023, 8, 31)

start_date = sl.date_input("Start date", min_value=min_date, max_value=max_date, value=min_date)
end_date = sl.date_input("End date", min_value=min_date, max_value=max_date, value=max_date)

if start_date <= end_date:
    sl.success("Start date: `{}`\n\nEnd date:`{}`".format(start_date, end_date))
else:
    sl.error("Error: End date must be after start date.")

stock_data = yf.download('AAPL', start=start_date, end=end_date)
stock_data.reset_index(inplace=True)

fig = go.Figure()
fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], name='Close'))
fig.update_layout(title=f"AAP: Stock Price")
sl.plotly_chart(fig)

selected_date = sl.date_input("Select a date", min_value=min_date, max_value=max_date, value=max_date)

print(selected_date)

print(type(selected_date))

if selected_date:
    
    df = pdr.get_data_yahoo('AAPL', start='2000-01-01', end='2023-09-01')

    sl.subheader(f'Predit')
    
    res = predict(selected_date)

    if (res == None):
        sl.write("data error")
    else:
        sl.write(f" the predict Close price is {res[0]} and actual price is {res[1]} there have distance {res[2]}")

