import dash
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.express as px
import numpy as np
import dash_html_components as html
import dash_core_components as dcc
import lasio
import math


def combination(n, r):
    """
    Function for finding combinations of a sequence
    @param: n.
    """
    return int((math.factorial(n)) / ((math.factorial(r)) * math.factorial(n - r)))

def pascals_triangle(rows):
    result = []
    for count in range(rows):
        row = []
        for element in range(count + 1):
            row.append(combination(count, element))
        result.append(row)
    return result[rows-1]

def binom_filter(x, kernel):
    """
    Function that applies the binomial filter
    @param x: 1D array on which to apply the filter
    @param kernel: 1D array with binomial kernel
    retrurn
    """
    return np.mean(np.convolve(x, kernel, 'same'))

infile = r"C:\Users\symin\github\borehole_gfx_utils\data\10WP32PB_induction.las"

las = lasio.read(infile)

df = las.df()

stylesheet = "https://codepen.io/chriddyp/pen/bWLwgP.css"
app = dash.Dash(__name__, external_stylesheets=[stylesheet])

log = 'INDUCTION_CALIBRATED'

df = df.dropna()
df = df[df[log] > 0.] * (1/1000.)
df = df.apply(np.log10)

depths = df.index.values

app.layout = html.Div([
    html.Div([
        html.Div(
                dcc.Graph(id = 'log-plot'),
            className = 'six columns'),
        html.Div(
                dcc.RangeSlider(
                            id='top_and_tail',
                            min=np.min(depths),
                            max=np.max(depths),
                            step=0.1,
                            value=[np.min(depths),np.max(depths)]
                        ), className = 'two columns'),
        ], className = "row"),
    html.Div([
        dcc.Slider(
            id='filter-size',
            min=1,
            max=99,
            step=2,
            value=33,
        )], className = "six columns"),
    ])


# Render log
@app.callback(Output('log-plot', 'figure'),
              [Input('top_and_tail', 'value'),
               Input('filter-size', 'value')])
def render_log(top_and_tail, filter_size):
    #fig = make_subplots(rows=1, cols=2, shared_yaxes=True,
    fig = px.line(df, x=log, title=log)
    fig.update_yaxes(autorange=True)


    return fig



if __name__ == '__main__':
    app.run_server(debug=True)