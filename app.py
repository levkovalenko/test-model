from pymongo import MongoClient
from collections import OrderedDict
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

import dash
from dash import  dcc, html
import plotly.express as px
from plotly import  graph_objects as go
import pandas as pd
from dash.dependencies import Input, Output



client = MongoClient('mongodb://theadmin:axcaxs@34.116.221.63:27017/Investments')
db = client.Investments
names = ['Cerner',
 'Kennametal Inc',
 'McDonald’s',
 'ONEOK',
 'Discovery Communications (C)',
 'Barrett Business Services Inc',
 'JD.com',
 'TCR2 Therapeutics Inc',
 'Select Energy Services Inc',
 'Quidel Corp',
 'China Southern Airlines',
 'Allegiant Travel Co',
 'Ingersoll-Rand',
 'Whirlpool',
 'Wyndham Hotels & Resorts'
 ]

df = pd.read_csv('filtered_300.csv')

def get_data(name):
    print(name)
    sc = MinMaxScaler(feature_range=(0,1))
    data = df[[name]]
    data = data.assign(scale=sc.fit_transform(data[name]))
    return data, sc


def load_model_from_file(path):
  return load_model(path)

def get_predict(model, time_serie, d, scaler):
  time_serie=np.array(time_serie)
  time_serie=time_serie.reshape(time_serie.shape[0],1)
  for _ in range(d):
    one_day = one_day_predict(model, time_serie).reshape(-1,1)
    time_serie = np.append(time_serie,one_day)
  ts_transformed = scaler.inverse_transform(time_serie.reshape(-1,1))
  return ts_transformed[-d:], ts_transformed

def one_day_predict(model, ser):
    data = []
    for j in range(15,len(ser)):
      data.append(ser[j-15:j])
    data = np.array(data)
    data = data[:, :, np.newaxis]
    return model.predict(data)[-1]




app = dash.Dash(__name__)

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options

app.layout = html.Div(children=[
    html.H1(children='Hello Invest Analysis'),
    
    dcc.Dropdown(
        id='name-dropdown',
        options=[
           {'label':name,'value':name} for name in names
        ],
        value='Cerner'
    ),

    dcc.Graph(
        id='predict-graph',
        figure=go.Figure()
    )
])

@app.callback(
    Output('predict-graph', 'figure'),
    Input('name-dropdown', 'value')
)
def update_output(name):
    data, sc = get_data(name)
    model = load_model_from_file('lstm_time_series.h5')
    base, predict = get_predict(model, data['scale'], scale=sc)
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=predict, name='predict'))
    return fig


if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')