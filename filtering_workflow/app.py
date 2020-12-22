import dash
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
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
    return np.array(result, dtype = object)[-1]

def binom_filter(x, kernel):
    """
    Function that applies the binomial filter
    @param x: 1D array on which to apply the filter
    @param kernel: 1D array with binomial kernel
    retrurn
    """
    return np.mean(np.convolve(x, kernel, 'valid'))

def run_filter(series, window, min_periods, filter_name):
    """
    Function that applies the function to the pandas series. Mostly based on
    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rolling.html
    @param series: pandas series with the to be filtered data
    @param: window: integer with filter window size
    @param: min_periods:minimum observation to have a value returned
    @param: filter: 'median', 'binomial' or 'mean': define what type of filter to use

    returns:
    a Window or Rolling sub-classed for the particular operation
    """
    if filter_name == 'median':
        return series.rolling(window=window, min_periods=min_periods).median()

    elif filter_name == 'binomial':
        kernel = pascals_triangle(window) / np.sum(pascals_triangle(window))
        return series.rolling(window=window, min_periods=min_periods).apply(binom_filter, args=(kernel,), raw=True)

    elif filter_name == 'mean':
        return series.rolling(window=window, min_periods=min_periods).mean()

def las2df(lasfile, logs):
    las = lasio.read(lasfile)
    df = las.df()
    # more processing steps
    dataframes = []
    ## TODO remove this shmozzle of an idea
    for log in logs:
        df_log = df[log].dropna()
        df_log = df_log[df_log > 0.] * (1 / 1000.) # convert to S/m
        df_log = df_log.apply(np.log10) # apply logarithmic transform
        dataframes.append(df_log)
    return dataframes

def row2filter_vars(df, row_number):
    # create a single row dataframe
    row = df.iloc[row_number]
    # get our values
    lasfile = row['induction_path']
    # now get the dataframe
    logs = row['Columns'].split(',')
    dataframes = las2df(lasfile, logs)
    filters = [row['filter1'], row['filter2']]
    minimum_windows = [row['filter1_min_window'], row['filter2_min_window']]
    filter_windows = [row['filter1_window_size'], row['filter2_window_size']]
    min_depth = row['min_depth']
    if np.isnan(min_depth):
        min_depth = np.min(dataframes[0].index)
    max_depth = row['max_depth']
    if np.isnan(max_depth):
        max_depth = np.max(dataframes[0].index)

    return {'dataframes': dataframes, 'filter_names': filters, 'filter_windows': filter_windows,
            'min_windows': minimum_windows, 'min_depth': min_depth, 'max_depth': max_depth,
            'logs': logs}

df_master = pd.read_csv('ind.csv')

# Get the intial filtering parameters
params = row2filter_vars(df_master, 0)

stylesheet = "https://codepen.io/chriddyp/pen/bWLwgP.css"
app = dash.Dash(__name__, external_stylesheets=[stylesheet])



app.layout = html.Div([
    html.Div([
        html.Div(
                dcc.Graph(id = 'log-plot', style = dict(height = 600)),
            className = 'six columns'),
        html.Div([
                dcc.RangeSlider(
                            id='top_and_tail',
                            min=np.min(params['dataframes'][0].index),
                            max=np.max(params['dataframes'][0].index),
                            step=0.1,
                            value=[params['min_depth'],params['max_depth']],
                            vertical=True,
                            verticalHeight=400,
                        ),
        html.Div(id='clipping-output', style={'margin-top': 20})],
            style={'marginTop':90},
            className = 'two columns'),
        ], className = "row"),
    html.Div([
        dcc.Slider(
            id='filter-size',
            min=9,
            max=55,
            step=1,
            value=params['filter_windows'][0],
        ),
    html.Div(id='filter-output', style={'margin-top': 20})
    ], className = "six columns"),
    dcc.Store(id='filtering_parameters'),
    ])


# Render log
@app.callback([Output('log-plot', 'figure'),
               Output('filter-output', 'children'),
               Output('clipping-output', 'children')],
              [Input('top_and_tail', 'value'),
               Input('filter-size', 'value')])
def render_log(top_and_tail, filter_window):
    fig = make_subplots(rows=1, cols=1)
    df_log = params['dataframes'][0]
    print(df_log)
    # Hack
    min_depth = np.max(df_log.index) - top_and_tail[1]
    max_depth = np.max(df_log.index) - top_and_tail[0]
    fig = fig.add_trace(go.Scatter(y=df_log.index, x=df_log.values, mode= "lines", name = "unfiltered"), row = 1, col = 1)
    fig.update_yaxes(autorange='reversed', row=1,col=1)
    # plot 2 will be the filtered trace
    depth_mask = (df_log.index > min_depth) & (df_log.index < max_depth)
    df_log.at[~depth_mask,log] = np.nan
    # To do replace global variable
    df_log['filtered'] = df_log[log].copy()
    for item in filter_names:
        df_log['filtered'] = run_filter(df_log['filtered'], filter_window, minimum_windows, item)

    fig.add_trace(go.Scatter(y=depths, x=df_log['filtered'].values, mode= "lines", name = "filtered"), row = 1, col = 1, )
    filter_msg = "Filter used were {} with a filter window of {}".format(', '.join(filter_names), str(filter_window))
    clipper_msg = "Data above {} m and below {} m depth have been clipped".format(str(np.round(min_depth, 2)), str(np.round(max_depth, 2)))

    return [fig, filter_msg, clipper_msg]

if __name__ == '__main__':
    app.run_server(debug=True)