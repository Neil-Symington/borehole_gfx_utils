'''
The purpose of this script is to generate useful plots of headworks for boreholes. The boreholes and construction info
should have been pulled from GA's oracle database using the extract_boreholes_data.py script

Neil Symington
neil.symington@ga.gov.au
'''

import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import geopandas as gpd

infile = r"C:\Users\u77932\Documents\github\borehole_gfx_utils\headworks_viz\AUS_2016_AUST.shp"

gdf = gpd.read_file(infile)


# assign the maximum and minimum relative heights of the boreh headworks and construction in the image coordinates

max_bore_height = 0.35
max_bore_depth = 0.35

def get_headworks_info(df_head):

    refs = {}

    for index, row in df_head.iterrows():
        key = row['DEPTH_REFERENCE_TYPE']
        # get values
        #convert to m
        if row['DEPTH_REFERENCE_UOM'] == 'mm':
            val = row['DEPTH_REFERENCE_HEIGHT']/1000.
        elif row['DEPTH_REFERENCE_UOM'] == 'm':
            val = row['DEPTH_REFERENCE_HEIGHT']
        else:
            print('please convert depth reference unit of measurements for borehole {} to m'.format(row['BOREHOLE_NAME']))
            return None
        # Get the elevation
        # convert to m
        #print(row)
        if row['ELEVATION_UOM'] == 'mm':
            elev = row['ELEVATION_VALUE']/ 1000.
        elif row['ELEVATION_UOM'] == 'm':
            elev = row['ELEVATION_VALUE']
        else:
            print('please convert datum elevation measurements for borehole {} to m'.format(
                row['BOREHOLE_NAME']))
            return None
        # elevatoin uncertainty
        if row['ELEVATION_UOM'] == 'mm':
            stdev = row['INPUT_UNCERTAINTY_Z'] / 1000.
        elif row['ELEVATION_UOM'] == 'm':
            stdev = row['INPUT_UNCERTAINTY_Z']

        # get datums
        elev_datum = row['ELEVATION_DATUM']
        depth_datum = row['DEPTH_REFERENCE_DATUM']
        refs[key] = {'local_height': val, 'depth_datum': depth_datum,
                     'ground_elevation': elev, 'datum': elev_datum, 'elev_stdev': stdev,
                     'DEPTH_REFERENCE_TYPE_ID': row['DEPTH_REFERENCE_TYPE_ID']}
    # create a dataframe
    df_refs = pd.DataFrame.from_dict(refs, orient = 'index')
    # now we want to have all local height datums relative to the ground
    assert len(df_refs['ground_elevation'].unique()) == 1
    assert len(df_refs['elev_stdev'].unique()) == 1
    for index, row in df_refs.iterrows():
        if row['depth_datum'].upper() == row['datum'].upper():
            df_refs.at[index, 'local_height'] = row['local_height'] - row['ground_elevation']
            df_refs.at[index,'depth_datum'] = 'ground level'

    return df_refs

def plot_borehole(df_head, df_constr = None):
    # Now add any construction information
    construction_plot_params = {'top of casing': {'linewidth': 1, 'edgecolor': 'k', 'facecolor': 'darkgrey',
                                           'width': 0.1, 'xmin': 0.45, 'label': 'casing'},
                                'top of casing protector': {'linewidth': 1, 'edgecolor': 'k', 'facecolor': 'white',
                                                     'width': 0.2, 'xmin': 0.4, 'label': 'casing protector'},
                                'concrete pad': {'linewidth': 2, 'edgecolor': 'k', 'facecolor': 'dimgray',
                                                 'width': 0.4, 'xmin': 0.3, 'label': 'concrete pad'},
                                'unknown': {'linewidth': 0.8, 'edgecolor': 'gray', 'facecolor': 'lightgray',
                                            'width': 0.06, 'xmin': 0.47, 'label': 'unknown'},
                                # casing protector and casing and screen are for the subsurface elements
                                'casing': {'linewidth': 1, 'edgecolor': 'k', 'facecolor': 'darkgrey',
                                                  'width': 0.1, 'xmin': 0.45},
                                'casing protector': {'linewidth': 1, 'edgecolor': 'k', 'facecolor': 'white',
                                                            'width': 0.2, 'xmin': 0.4},
                                'screen': {'linewidth': 1, 'edgecolor': 'b', 'facecolor': 'cornflowerblue',
                                           'width': 0.14, 'xmin': 0.43, 'label': 'screen'},
                                }


    fig, ax = plt.subplots(1,1, figsize = (8.3, 11.7))

    df_refs = get_headworks_info(df_head).sort_values(by = 'local_height')


    # Now we want to assign the relative heights for each construction element by sorting them
    df_refs['rel_height'] = (df_refs['local_height'] / df_refs['local_height'].max()) * max_bore_height

    for index, row in df_refs.iterrows():

        if index in construction_plot_params.keys():
            construction_plot_params[index]['headworks_height'] = row['rel_height']

    # iteratively add all elements
    for item in ['concrete pad', 'top of casing protector', 'top of casing', 'unknown']:
        if not item in df_refs.index:
            continue

        params = construction_plot_params[item]
        ax.add_patch(Rectangle((params['xmin'], 0.5), params['width'], params['headworks_height'],
                               linewidth=params['linewidth'], edgecolor=params['edgecolor'],
                               facecolor=params['facecolor'], label=params['label']))
        # add reference lines
        plt.axhline(y=0.5 + params['headworks_height'],
                    xmin=0.25, xmax=0.5 - params['width']/2., c='grey', linestyle='dashed', linewidth=0.5)

        plt.axhline(y=0.5 + params['headworks_height'],
                    xmin=0.5 + params['width']/2., xmax=0.75, c='grey', linestyle='dashed', linewidth=0.5)
        # add text
        text = ' '.join([str(np.round(df_refs.loc[item, 'local_height'], 3)), 'mAGL'])
        ax.text(x=0.1, y=0.5 + params['headworks_height'], s=text)

    # Now we want to add distances between the items
    if len(df_refs) > 1:
        for i in range(len(df_refs) - 1):
            pos0 = df_refs.iloc[i]['rel_height'] + 0.5
            rel_offset = df_refs.iloc[i+1]['rel_height'] - df_refs.iloc[i]['rel_height']
            distance = df_refs.iloc[i+1]['local_height'] - df_refs.iloc[i]['local_height']
            # plot arrows
            if distance > 0.01:
                ax.annotate(text = '', xy = (0.73, pos0 + 0.001), xytext = (0.73, pos0 + rel_offset - 0.001),
                            arrowprops=dict(arrowstyle='<->'))
            # Now add the label
            ax.text(x= 0.75, y = pos0 + rel_offset/2., s = ' '.join([str(np.round(distance, 3)),
                                                                    'm']))

    # plot the borehole
    ax.add_patch(Rectangle((0.45, 0.5), 0.1, -0.35, linewidth=1, edgecolor='gray', facecolor='lightgray'))
    plt.axhline(y=0.15, xmin=0.3, xmax=0.45, c='grey', linestyle='dashed', linewidth=0.5)
    preferred_depth = np.round(df_head['PREFERRED_TD_M'].unique()[0], 1)
    b_text = ' '.join([str(-1*preferred_depth), 'mAGL'])
    ax.text(x = 0.1, y = 0.15, s = b_text)


    # if there is construction, add to the plot
    if df_constr is not None:
        # first we need to get the construction relative to ground level
        assert len(df_constr['DEPTH_REFERENCE_TYPE_ID'].unique() == 1)
        df_constr_merged = df_constr.merge(df_refs, on = 'DEPTH_REFERENCE_TYPE_ID')
        depth_from = df_constr_merged['INTERVAL_BEGIN'] + df_constr_merged['local_height']
        depth_from[depth_from < 0] = 0.
        # make sure depth from is never negative
        df_constr_merged['DEPTH_FROM'] = depth_from
        df_constr_merged['DEPTH_TO'] = df_constr_merged['INTERVAL_END'] + df_constr_merged['local_height']

        # get the relative depth to and height
        df_constr_merged['REL_INTERVAL_HEIGHT'] = max_bore_depth * (df_constr_merged['DEPTH_TO'] - df_constr_merged['DEPTH_FROM'])/preferred_depth
        df_constr_merged['REL_DEPTH_TO'] = 0.5 - df_constr_merged['DEPTH_TO']/preferred_depth * max_bore_depth


        # No iterate through and plot theelements
        for index, row in df_constr_merged.iterrows():
            construction_name = row['CONSTRUCTION_NAME']
            if construction_name in construction_plot_params:
                params = construction_plot_params[construction_name]
                rel_height =  row['REL_INTERVAL_HEIGHT']
                rel_depth_to = row['REL_DEPTH_TO']

                if construction_name in ['casing', 'casing protector']:
                    ax.add_patch(Rectangle((params['xmin'], rel_depth_to), params['width'],rel_height,
                                           linewidth=params['linewidth'], edgecolor=params['edgecolor'],
                                           facecolor=params['facecolor']))

                elif construction_name == 'screen':
                    ax.add_patch(Rectangle((params['xmin'], rel_depth_to), params['width'], rel_height,
                                           linewidth=params['linewidth'], edgecolor=params['edgecolor'],
                                           facecolor=params['facecolor'], label = params['label']))
                    # add reference lines
                    plt.axhline(y=rel_depth_to,
                                xmin=0.25, xmax=0.5 - params['width'] / 2., c='b', linestyle='dashed', linewidth=0.5)

                    plt.axhline(y=rel_depth_to + rel_height,
                                xmin=0.25, xmax=0.5 - params['width'] / 2., c='b', linestyle='dashed', linewidth=0.5)
                    # add text
                    text = ' '.join([str(-1*np.round(row['DEPTH_FROM'], 3)), 'mAGL'])
                    ax.text(x=0.1, y=rel_depth_to + rel_height, s=text)

                    text = ' '.join([str(-1*np.round(row['DEPTH_TO'], 3)), 'mAGL'])
                    ax.text(x=0.1, y=rel_depth_to, s=text)

    # plot ground level
    plt.axhline(y=0.5, xmin=0.25, xmax=0.75, c='green', label="ground")

    ground_text = ' '.join([str(np.round(df_refs.loc['ground surface', 'ground_elevation'], 2)),
                            r'$\pm$',
                            str(np.round(df_refs.loc['ground surface', 'elev_stdev'],3)), 'm'])
    if df_refs.loc['ground surface', 'datum'].upper() == 'AUSTRALIAN HEIGHT DATUM':
        ground_text+= "AHD"
    elif df_refs.loc['ground surface', 'datum'].upper() == 'GPS ELLIPSOID':
        ground_text += "\n above GPS ellipsoid"
    else:
        ground_text += '\nabove datum'

    ax.text(x=0., y=0.5, s=ground_text)

    ax.legend(loc = 2)
    # add the reference datum
    ax.text(y = 0.95, x = 0.4, s = df_head.iloc[0]['BOREHOLE_NAME'], fontsize = 16, weight  = 'bold')
    ax.text(y=0.9, x=0.4, s = ' '.join(["ENO:", str(df_head.iloc[0]['ENO'])]),
            fontsize=10)
    # Now lets add the coordinates
    easting, northing = df_head.iloc[0][['easting', 'northing']].values
    projected_crs = df_head.iloc[0]['projected_crs']

    GDA94_LONGITUDE, GDA94_LATITUDE = df_head.iloc[0][['GDA94_LONGITUDE', 'GDA94_LATITUDE']].values
    geographic_crs = 'GDA94'

    ax.text(y=0.35, x=0.67, s=' '.join([projected_crs, ':']), fontsize = 10, weight = 'bold')
    ax.text(y=0.32, x=0.7, s=' '.join(['easting:',
                                      str(np.round(easting, 1))]))
    ax.text(y=0.3, x=0.7, s=' '.join(['northing:',
                                      str(np.round(northing, 1))]))
    # add the
    ax.text(y=0.27, x=0.67, s=' '.join([geographic_crs, ':']), fontsize=10, weight='bold')
    ax.text(y=0.24, x=0.7, s=' '.join(['longitude:',
                                       str(np.round(GDA94_LONGITUDE, 6))]))
    ax.text(y=0.22, x=0.7, s=' '.join(['latitude:',
                                       str(np.round(GDA94_LATITUDE, 6))]))
    return fig, ax



# read in borehole information
infile = "EFTF_header.csv"

df_header = pd.read_csv(infile)

infile = "EFTF_construction.csv"

df_construction = pd.read_csv(infile)

enos = df_header['ENO'].unique()

for e in enos:
    print(e)

    # create a subset of header and construction
    df_header_ss = df_header[df_header['ENO'] == e]
    # construction
    df_construction_ss = df_construction[df_construction['BOREHOLE_ID'] == e]

    if len(df_construction_ss) > 0:
        fig, ax = plot_borehole(df_header_ss, df_construction_ss)
    else:
        fig, ax = plot_borehole(df_header_ss)

    # add the map of Australia

    ax.axis("off")
    ax2 = fig.add_axes([0.6, 0.7, 0.28, 0.28])
    gdf.plot(ax = ax2)
    ax2.scatter(df_header_ss.iloc[0]['GDA94_LONGITUDE'],
                df_header_ss.iloc[0]['GDA94_LATITUDE'], c= 'red', s = 6.)
    ax2.axis("off")

    outfile = os.path.join(r"C:\Temp\headworks_figures", '.'.join([df_header_ss.iloc[0]['BOREHOLE_NAME'], 'png']))
    plt.savefig(outfile)
    plt.close('all')

