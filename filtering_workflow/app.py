import dash
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import pandas as pd
import numpy as np
import dash_html_components as html
import dash_core_components as dcc
import dash_table
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
        return series.rolling(window=window, min_periods=min_periods, center = True).median()

    elif filter_name == 'binomial':
        kernel = pascals_triangle(window) / np.sum(pascals_triangle(window))
        return series.rolling(window=window, min_periods=min_periods,
                              center = True).apply(binom_filter, args=(kernel,), raw=True)

    elif filter_name == 'mean':
        return series.rolling(window=window, min_periods=min_periods, center = True).mean()

def lasfile2df(lasfile):
    df = lasio.read(lasfile).df()
    df = df.dropna(how = 'all')
    df = df * (1 / 1000.) # convert to S/m
    df = df.apply(np.log10) # apply logarithmic transform
    return df


def row2filter_vars(df, row_number):
    # create a single row dataframe
    row = df.iloc[row_number]
    # get our values
    lasfile = row['induction_path']
    # now get the log names
    logs = row['Columns'].split(',')
    # create an options dictionary for the dropdownmenu
    log_options = []
    for log in logs:
        log_options.append({'label': log, 'value': log})
    filters = [row['filter1'], row['filter2']]
    minimum_windows = [row['filter1_min_window'], row['filter2_min_window']]
    filter_windows = [row['filter1_window_size'], row['filter2_window_size']]
    resampling_interval = row['sampling_interval']

    df = lasfile2df(lasfile).dropna(how='any')
    min_depth = np.min(df.index)
    max_depth = np.max(df.index)
    data = df.to_dict('split')
    WELL = row['WELL']

    return {'lasfile': lasfile, 'filter_names': filters, 'filter_windows': filter_windows,
            'min_windows': minimum_windows, 'min_depth': min_depth, 'max_depth': max_depth,
            'logs': logs, 'resampling_interval': resampling_interval, 'log_options': log_options,
            'data' : data, 'well': WELL}

def find_trigger():
    ctx = dash.callback_context
    # Find which input triggered the callback
    if ctx.triggered:
        trig_id = ctx.triggered[0]['prop_id']
    else:
        trig_id = None
    return trig_id

df_master = pd.read_csv('EFTF_induction_filtering_master_.csv')


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
                            step=0.1,
                            vertical=True,
                            verticalHeight=400,
                        ),
        html.Div(id='clipping-output', style={'margin-top': 20})],
            style={'marginTop':90},
            className = 'two columns'),
        html.Div([dcc.Dropdown(id = "log-dropdown",
                               options=params['log_options'],
                               value=(params['logs'][0])),
                  dash_table.DataTable(id='master_table',
                                css=[{'selector': '.row', 'rule': 'margin: 0'}],
                                columns = [{"name": i, "id": i} for i in df_master.columns],
                                data=df_master.to_dict('records'),
                                fixed_columns={ 'headers': True},
                                sort_action="native",
                                sort_mode="multi",
                                row_selectable="single",
                                row_deletable=True,
                                selected_columns=[],
                                selected_rows=[0],
                                style_header={'backgroundColor': 'rgb(30, 30, 30)',
                                             'height': '40px'},
                                style_cell={
                                             'backgroundColor': 'rgb(50, 50, 50)',
                                             'color': 'white',
                                             'minHeight': '50px',
                                             'minWidth': '0px', 'maxWidth': '80px',
                                             'whiteSpace': 'normal',
                                             'font-size': '10px'
                                         },
                              style_table={
                                          'maxHeight': '400px',
                                          'overflowY': 'scroll',
                                          'maxWidth':  '90%',
                                          'overflowX': 'scroll'}),
                    html.Button('Export_log', id='export-button', n_clicks=0),
                    html.Div(id='export-output')
                  ],style={'marginTop':90},className = 'four columns')],
        className = "row"),
    html.Div([
        dcc.Slider(
            id='filter-size_1',
            min=5,
            max=55,
            step=1,
            value=params['filter_windows'][0],
        ),
dcc.Slider(
            id='filter-size_2',
            min=5,
            max=55,
            step=1,
            value=params['filter_windows'][1],
        ),
    html.Div(id='filter-output', style={'margin-top': 20})
    ], className = "six columns"),
    dcc.Store(id='filtering_parameters'),
    dcc.Store(id='filtered_logs'),
    dcc.Store(id='raw_logs')
    ])


# Processing data call back
@app.callback(Output('filtering_parameters', 'data'),
              [Input('master_table', "derived_virtual_selected_rows")])
def process_row(selected_rows):

    selected_rows = selected_rows or [0]

    params = row2filter_vars(df_master, selected_rows[0])

    return params

# Processing data call back
@app.callback(Output('export-output', 'children'),
              [Input('export-button', "n_clicks")],
              [State('filtered_logs', 'data')])
def export_results(nclicks, filtered_logs):

    if nclicks > 0:
        filt = filtered_logs['filtered_data']
        df_las = pd.DataFrame(index=filt['index'], data=filt['data'],
                              columns=filt['columns'])
        df_las.to_csv(r"C:\temp\example.txt")
        return "Export successful"
    else:
        return ""

# Render log
@app.callback([Output('log-plot', 'figure'),
               Output('filter-output', 'children'),
               Output('clipping-output', 'children'),
               Output('filtered_logs', 'data')],
              [Input('top_and_tail', 'value'),
               Input('filter-size_1', 'value'),
               Input('filter-size_2', 'value'),
               Input('log-dropdown', 'value'),
               Input('filtering_parameters', 'data')],
               [State('master_table', "derived_virtual_selected_rows")])
def render_log(top_and_tail, filter_window_1, filter_window_2, log, params, selected_rows):
    trig_id = find_trigger()

    selected_rows = selected_rows or [0]

    params = row2filter_vars(df_master, selected_rows[0])

    raw_logs = params['data']

    df_las = pd.DataFrame(index = raw_logs['index'], data = raw_logs['data'],
                          columns = raw_logs['columns']).dropna(how='all')

    if trig_id == "log-dropdown.value":
        log = log
    else:
        log = params['logs'][0]

    params['filter_windows'][0] = filter_window_1

    params['filter_windows'][1] = filter_window_2

    if trig_id == 'top_and_tail.value':
        top_and_tail = top_and_tail
    else:
        top_and_tail = [0., params['max_depth']]

    filter_windows = params['filter_windows']
    fig = make_subplots(rows=1, cols=1)

    depths = df_las.index.values
    values = df_las[log]

    # Hack
    min_depth = params['max_depth'] - top_and_tail[1]
    max_depth = params['max_depth'] - top_and_tail[0]

    fig = fig.add_trace(go.Scatter(y=depths,x=values, mode= "lines", name = log + " unfiltered"), row = 1, col = 1)
    fig.update_yaxes(autorange='reversed', row=1,col=1)

    df_las['filtered'] = np.nan

    for i, item in enumerate(params['filter_names']):
        df_las['filtered'] = run_filter(df_las[log], filter_windows[i],
                                        params['min_windows'][i], item)

    # plot 2 will be the filtered trace
    depth_mask = (depths > min_depth) & (depths < max_depth)

    df_las.at[~depth_mask, 'filtered'] = np.nan

    # Finally we resample
    sampling_interval = stats.mode(depths[1:] - depths[:1])[0][0]
    # now find the resamping factor
    n = np.ceil(params['resampling_interval'] / sampling_interval).astype(np.int)

    df_downsampled = df_las.dropna().iloc[::n]

    fig.add_trace(go.Scatter(y=df_downsampled.index, x=df_downsampled['filtered'].values,
                             mode= "lines", name = log + " filtered"),
                  row = 1, col = 1, )

    filter_msg = "Filter used were {} with a filter window of {}".format(', '.join(params['filter_names']),
                                                                         ', '.join([str(x) for x in filter_windows]))

    clipper_msg = "Data above {} m and below {} m depth have been clipped".format(str(np.round(min_depth, 2)),
                                                                                  str(np.round(max_depth, 2)))

    filtered_output = {'filtered_data': df_downsampled.to_dict('split'), 'min_depth': min_depth,
                       'max_depth': max_depth, 'log': log, 'filter_names': params['filter_names'],
                       'filter_windows': filter_windows}

    return [fig, filter_msg, clipper_msg, filtered_output]

# update drop down menu
@app.callback([Output('log-dropdown', 'value'),
               Output('log-dropdown', 'options')],
              [Input('master_table', "derived_virtual_selected_rows")])
def update_dropdown(selected_rows):
    selected_rows = selected_rows or [0]

    log_options = row2filter_vars(df_master, selected_rows[0])['log_options']

    return log_options[0]['value'], log_options

# update drop down menu
@app.callback([Output('top_and_tail', 'min'),
               Output('top_and_tail', 'max'),
               Output('top_and_tail', 'value')],
              [Input('filtering_parameters', 'data')],
               [State('master_table', "derived_virtual_selected_rows")])
def update_slider(params, selected_rows):

    print(params['well'])

    raw_logs = params['data']

    df_las = pd.DataFrame(index=raw_logs['index'], data=raw_logs['data'], columns=raw_logs['columns']).dropna(how='all')

    value = [params['min_depth'], params['max_depth']]
    min = 0.
    max = np.max(df_las.index)

    print(value)

    return [min, max, value]


if __name__ == '__main__':
    app.run_server(debug=True)