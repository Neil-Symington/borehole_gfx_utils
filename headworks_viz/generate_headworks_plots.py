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
                     'ground_elevation': elev, 'datum': elev_datum, 'elev_stdev': stdev}
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

def plot_headworks(df_head, df_contr):
    fig, ax = plt.subplots(1,1, figsize = (8.3, 11.7))
    # Start by plotting the casing protector
    df_refs = get_headworks_info(df_head)
    # plot ground level
    plt.axhline(y=0.5, xmin = 0.3, xmax = 0.7, c='green', label="ground")

    ground_text = ' '.join([str(np.round(df_refs.loc['ground surface', 'ground_elevation'],2)),
                            r'$\pm$',
                            str(df_refs.loc['ground surface', 'elev_stdev']),'m'])
    ax.text(x = 0.73, y = 0.5, s= ground_text)

    #start with the casing protector
    if 'top of casing protector' in df_refs.index:
        casing_protector_rel_height = 0.35
        if 'top of casing' in df_refs.index:
            if df_refs.loc['top of casing protector', 'local_height'] < df_refs.loc['top of casing', 'local_height']:
                ratio = df_refs.loc['top of casing protector', 'local_height'] / df_refs.loc['top of casing', 'local_height']
                casing_protector_rel_height *= ratio
        else:
            pass

        # add a reference lines
        plt.axhline(y=0.5 + casing_protector_rel_height,
                    xmin=0.6, xmax=0.7, c='grey', linestyle = 'dashed', linewidth = 0.5)
        plt.axhline(y=0.5 + casing_protector_rel_height,
                    xmin=0.3, xmax=0.4, c='grey', linestyle='dashed', linewidth=0.5)
        ax.add_patch(Rectangle((0.4, 0.5), 0.2, casing_protector_rel_height,
                              linewidth=1, edgecolor='k', facecolor='white',
                              label = 'casing protector'))
        casing_protector_elevation = df_refs.loc['top of casing protector',
                                                 'local_height'] + df_refs.loc['ground surface',  'ground_elevation']
        # add text
        cp_text1 = ' '.join([str(np.round(df_refs.loc['top of casing protector',
                                                 'local_height'], 3)), 'mAGL'])
        cp_text2 = ' '.join([str(np.round(casing_protector_elevation, 2)), 'm'])
        ax.text(x = 0.1, y= 0.5 + casing_protector_rel_height, s=cp_text1)
        ax.text(x = 0.73, y = 0.5 + casing_protector_rel_height, s = cp_text2)

    if 'top of casing' in df_refs.index:
        # get ratio of casing protector to casing
        if 'top of casing protector' in df_refs.index:
            ratio = df_refs.loc['top of casing','local_height'] / df_refs.loc['top of casing protector','local_height']
            casing_rel_height = casing_protector_rel_height * ratio

        else:
            casing_rel_height = 0.35

        casing_elevation = df_refs.loc['top of casing', 'local_height'] + df_refs.loc[
            'ground surface', 'ground_elevation']
        # add text
        c_text1 = ' '.join([str(np.round(df_refs.loc['top of casing',
                                                     'local_height'], 3)), 'mAGL'])
        c_text2 = ' '.join([str(np.round(casing_elevation, 2)), 'm'])
        ax.text(x=0.1, y=0.5 + casing_rel_height, s=c_text1)
        ax.text(x=0.73, y=0.5 + casing_rel_height, s=c_text2)
        # add a reference lines
        plt.axhline(y=0.5 + casing_rel_height,
                    xmin=0.6, xmax=0.7, c='grey', linestyle='dashed', linewidth=0.5)
        plt.axhline(y=0.5 + casing_rel_height,
                    xmin=0.3, xmax=0.4, c='grey', linestyle='dashed', linewidth=0.5)
        ax.add_patch(Rectangle((0.45, 0.5), 0.1, casing_rel_height, linewidth=1, edgecolor='k', facecolor='grey',
                              label = 'casing'))
    ax.legend(loc = 2)
    # add the reference datum
    datum_text = 'elevation relative to \n{}'.format(df_refs.loc['ground surface', 'datum'])
    ax.text(y = 0.4, x = 0.7, s = datum_text)
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
    ax.text(y=0.27, x=0.65, s=' '.join([geographic_crs, ':']), fontsize=10, weight='bold')
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

    # create a subset of header and construction
    df_header_ss = df_header[df_header['ENO'] == e]
    # construction
    df_construction_ss = df_construction[df_construction['BOREHOLE_ID'] == e]
    if len(df_construction_ss) > 0:
        fig, ax = plot_headworks(df_header_ss, df_construction_ss)
    else:
        fig, ax = plot_headworks(df_header_ss, df_construction_ss)
    ax.axis("off")
    ax2 = fig.add_axes([0.15, 0.1, 0.3, 0.2])
    gdf.plot(ax = ax2)
    ax2.scatter(df_header_ss.iloc[0]['GDA94_LONGITUDE'],
             df_header_ss.iloc[0]['GDA94_LATITUDE'], c= 'red')

    outfile = os.path.join(r"C:\Temp\headworks_figures", '.'.join([df_header_ss.iloc[0]['BOREHOLE_NAME'], 'png']))
    plt.savefig(outfile)
    plt.close('all')

