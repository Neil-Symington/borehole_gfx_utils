import os
import dash
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.io import write_html
from scipy import stats
import pandas as pd
import numpy as np
import dash_html_components as html
import dash_core_components as dcc
import dash_table
import lasio
import math
import time

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


def row2filter_vars(df, row_number, get_frame = True):
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

    WELL = row['WELL']
    params = {'lasfile': lasfile, 'filter_names': filters, 'filter_windows': filter_windows,
                'min_windows': minimum_windows, 'min_depth': min_depth, 'max_depth': max_depth,
                'logs': logs, 'resampling_interval': resampling_interval, 'log_options': log_options,
                'well': WELL}
    if get_frame:
        data = df.to_dict('split')
        return data, params
    else:
        return params


def find_trigger():
    ctx = dash.callback_context
    # Find which input triggered the callback
    if ctx.triggered:
        trig_id = ctx.triggered[0]['prop_id']
    else:
        trig_id = None
    return trig_id

df_master = pd.read_csv('EFTF_induction_filtering_master_.csv')

outdir = r"\\prod.lan\active\proj\futurex\Common\Working\Neil\filtered_borehoel_gfx\induction"

if not os.path.exists(outdir):
    os.mkdir(outdir)

# Get the intial filtering parameters
init_data, init_params = row2filter_vars(df_master, 0, get_frame =True)

init_depth = init_data['index']

stylesheet = "https://codepen.io/chriddyp/pen/bWLwgP.css"
app = dash.Dash(__name__, external_stylesheets=[stylesheet])

app.layout = html.Div([
    html.Div([
        html.Div(
                dcc.Graph(id = 'log-plot', style = dict(height = 600)),
            className = 'six columns'),
        html.Div([
            html.Div(["Minimum depth : ", dcc.Input(
                        id="min_depth", type="number",
                        min=0., max=np.max(init_depth) - 0.1,
                value = init_params['min_depth'])], style={'marginTop':40 },
                     className='row'),
            html.Div(["Maximum depth: ", dcc.Input(
                    id="max_depth", type="number",
                    min=np.min(init_depth), max=np.max(init_depth),
                value=init_params['max_depth'])],  style={'marginTop':200},
                     className='row'),
        html.Div(id='clipping-output', style={'margin-top': 20})],
            style={'marginTop':90},
            className = 'two columns'),
        html.Div([dcc.Dropdown(id = "log-dropdown",
                               options=init_params['log_options'],
                               value=(init_params['logs'][0])),
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
            value=init_params['filter_windows'][0],
        ),
        dcc.Slider(
            id='filter-size_2',
            min=5,
            max=55,
            step=1,
            value=init_params['filter_windows'][1],
        ),
        html.Div(["Filter 1 minimum window: ", dcc.Input(
            id="min_window_1", type="number",
            min=1, max=33, value=5)],
                           className='row'),
        html.Div(["Filter 2 minimum window: ", dcc.Input(
                    id="min_window_2", type="number",
                      min=1, max=33, value=5)],
                           className='row'),

        html.Div(id='filter-output', style={'margin-top': 20})
    ], className = "six columns"),
    dcc.Store(id='filtering_params'),
    dcc.Store(id='filtered_logs'),
    dcc.Store(id='raw_logs'),
    dcc.Store(id = 'initial_params')
    ])


# Processing data call back
@app.callback([Output('initial_params', 'data'),
               Output('raw_logs', 'data'),
               Output('min_depth', 'min'),
               Output('min_depth', 'max'),
               Output('min_depth', 'value'),
               Output('max_depth', 'min'),
               Output('max_depth', 'max'),
               Output('max_depth', 'value'),
                Output('log-dropdown', 'value'),
               Output('log-dropdown', 'options')
               ],
              [Input('master_table', "derived_virtual_selected_rows")])
def process_row(selected_rows):

    selected_rows = selected_rows or [0]

    data, params = row2filter_vars(df_master, selected_rows[0])

    depths = data['index']
    # parse the data to get the min max and value

    min = np.min(depths)
    max = np.max(depths)

    min_depth_values = params['min_depth']
    max_depth_values = params['max_depth']

    log_options = params['log_options']

    return params, data, 0., max - 0.1, min_depth_values, min, max, max_depth_values, log_options[0]['value'], log_options

# Processing data call back
@app.callback(Output('export-output', 'children'),
              [Input('export-button', "n_clicks")],
              [State('filtered_logs', 'data'),
               State('filtering_params', 'data'),
               State('log-plot', 'figure'),
               State('log-dropdown', 'value')])
def export_results(nclicks, filtered_logs, filtered_params, log_plot, log):

    if nclicks > 0:
        tic = time.time()
        well_name = filtered_params['well']
        df_las = pd.DataFrame(index=filtered_logs['index'], data=filtered_logs['data'],
                              columns=filtered_logs['columns'])
        df_las['filtered_S/m'] = 10**df_las['filtered'].values
        # Write the log to a csv
        df_las['filtered_S/m'].to_csv(os.path.join(outdir, '_'.join([well_name, log, 'filtered.csv'])))
        # Write the image to a html
        write_html(log_plot,os.path.join(outdir, '_'.join([well_name, log, 'filtered.html'])))
        # Write the metadata to a file
        metadata = {'well': well_name, 'original_file': filtered_params['lasfile'],
                    'filters': ','.join([str(n) for n in filtered_params['filter_names']]),
                    'filter_windows': ','.join([str(n) for n in filtered_params['filter_windows']]),
                    'min_windows': ','.join([str(n) for n in filtered_params['min_windows']]),
                    'min_depth': filtered_params['min_depth'], 'max_depth': filtered_params['max_depth']}
        pd.DataFrame(metadata, index = [0]).to_csv(os.path.join(outdir, '_'.join([well_name, log, 'metadata.csv'])), index = False)
        toc = time.time()

        return "Export successfully in " + str(np.round(toc - tic,1)) + " seconds"
    else:
        return ""

# Render log
@app.callback([Output('log-plot', 'figure'),
               Output('filter-output', 'children'),
               Output('clipping-output', 'children'),
               Output('filtering_params', 'data'),
               Output('filtered_logs', 'data')],
              [Input("min_depth", 'value'),
               Input("max_depth", 'value'),
               Input('filter-size_1', 'value'),
               Input('filter-size_2', 'value'),
               Input('log-dropdown', 'value'),
               Input("min_window_1", 'value'),
               Input("min_window_2", 'value'),
               Input('initial_params', 'data')],
               [State('filtering_params', 'data'),
                State('raw_logs', 'data')])
def render_log(min_depth, max_depth, filter_window_1, filter_window_2, log, min_window_1, min_window_2, initial_params, filtering_params,
               raw_logs):
    trig_id = find_trigger()
    print(trig_id)
    # For none initialisation
    if raw_logs is None:

        raw_logs, initial_params = row2filter_vars(df_master, 0)

    df_las = pd.DataFrame(index = raw_logs['index'], data = raw_logs['data'],
                          columns = raw_logs['columns']).dropna(how='all')

    # We use the filtered params if we have already adjusted the inital params
    if trig_id == 'initial_params.data' or filtering_params is None:
        params = initial_params.copy()
    else:
        params = filtering_params.copy()

    if trig_id == "log-dropdown.value":
        log = log
    else:
        log = params['logs'][0]
    # Update our params with the call back values. These should always be
    params['filter_windows'][0] = filter_window_1

    params['filter_windows'][1] = filter_window_2

    params['min_windows'][0] = min_window_1

    params['min_windows'][1] = min_window_2

    filter_windows = params['filter_windows']
    fig = make_subplots(rows=1, cols=1)

    depths = df_las.index.values

    values = 10**df_las[log]

    fig = fig.add_trace(go.Scatter(y=depths,x=values, mode= "lines", name = log + " unfiltered"), row = 1, col = 1)
    fig.update_yaxes(autorange='reversed', row=1,col=1)

    df_las['filtered'] = df_las[log].copy()

    for i, item in enumerate(params['filter_names']):
        df_las['filtered'] = run_filter(df_las['filtered'], filter_windows[i],
                                        params['min_windows'][i], item)

    # plot 2 will be the filtered trace
    depth_mask = (depths > min_depth) & (depths < max_depth)

    df_las.at[~depth_mask, 'filtered'] = np.nan

    # Finally we resample
    sampling_interval = stats.mode(depths[1:] - depths[:1])[0][0]
    # now find the resamping factor
    n = np.ceil(params['resampling_interval'] / sampling_interval).astype(np.int)

    df_downsampled = df_las.dropna().iloc[::n]

    filtered_values = 10**df_downsampled['filtered'].values

    fig.add_trace(go.Scatter(y=df_downsampled.index, x=filtered_values,
                             mode= "lines", name = log + " filtered"),
                  row = 1, col = 1, )
    fig.update_xaxes(type="log", title_text= "Conductivity (S/m)")
    fig.update_yaxes(title_text="depth (m below reference datum)")
    fig['layout'].update({'uirevision': initial_params})

    filter_msg = "Filter used were {} with a filter window of {}".format(', '.join(params['filter_names']),
                                                                         ', '.join([str(x) for x in filter_windows]))

    clipper_msg = "Data above {} m and below {} m depth have been clipped".format(str(np.round(min_depth, 2)),
                                                                                  str(np.round(max_depth, 2)))

    filtered_params = {'lasfile': params['lasfile'], 'min_depth': min_depth, 'min_windows': params['min_windows'],
                       'max_depth': max_depth, 'log': log, 'filter_names': params['filter_names'],
                       'filter_windows': filter_windows, 'well': params['well'], 'logs': params['logs'],
                       'resampling_interval': params['resampling_interval']}

    filtered_logs = df_downsampled.to_dict('split')

    return [fig, filter_msg, clipper_msg, filtered_params, filtered_logs]



if __name__ == '__main__':
    app.run_server(debug=False)