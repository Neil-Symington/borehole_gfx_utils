import dash
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    Function that applies the functoin to the pandas series. Mostly based on
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



infile = r"E:\GA\induction_gamma\EK\2017_tfd_files_GA_ALT_system\17BP05I\17BP05I_induction_down.las"
filter_names = ['median', 'binomial']
minimum_windows = 9
log = 'DEEP_INDUCTION'

las = lasio.read(infile)

df = las.df()
depths = df.index.values

stylesheet = "https://codepen.io/chriddyp/pen/bWLwgP.css"
app = dash.Dash(__name__, external_stylesheets=[stylesheet])



df = df.dropna()
df = df[df[log] > 0.] * (1/1000.)
df = df.apply(np.log10)



app.layout = html.Div([
    html.Div([
        html.Div(
                dcc.Graph(id = 'log-plot', style = dict(height = 600)),
            className = 'six columns'),
        html.Div([
                dcc.RangeSlider(
                            id='top_and_tail',
                            min=np.min(depths),
                            max=np.max(depths),
                            step=0.1,
                            value=[np.min(depths),np.max(depths)],
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
            min=minimum_windows,
            max=55,
            step=1,
            value=15,
        ),
    html.Div(id='filter-output', style={'margin-top': 20})
    ], className = "six columns"),
    ])


# Render log
@app.callback([Output('log-plot', 'figure'),
               Output('filter-output', 'children'),
               Output('clipping-output', 'children')],
              [Input('top_and_tail', 'value'),
               Input('filter-size', 'value')])
def render_log(top_and_tail, filter_window):
    fig = make_subplots(rows=1, cols=1)
    df_log = df.copy()
    # Hack
    min_depth = np.max(depths) - top_and_tail[1]
    max_depth = np.max(depths) - top_and_tail[0]
    fig = fig.add_trace(go.Scatter(y=depths, x=df[log].values, mode= "lines", name = "unfiltered"), row = 1, col = 1)
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
    clipper_msg = "Data above {} m and below {} m depth have bee clipped".format(str(np.round(min_depth, 2)), str(np.round(max_depth, 2)))

    return [fig, filter_msg, clipper_msg]

if __name__ == '__main__':
    app.run_server(debug=True)