import os
import pandas as pd
import numpy as np
import sys
sys.path.append(r'H:\scripts')
import connect_to_oracle
from pyproj import CRS, Transformer

# Set up a connection to the oracle database
ora_con = connect_to_oracle.connect_to_oracle()

# Load the master spreadsheet

infile = "EFTF_induction_master.csv"
df_master = pd.read_csv(infile, keep_default_na=False)

enos = df_master['UWI'].values
drt = np.unique(df_master['DEPTH_REFERENCE_ID'].values)

# Create a sql query

header_query = """

select
   b.borehole_name,
   sum.ENO,
   sum.COUNTRY,
   sum.STATE,
   sum.ELEVATION_VALUE,
   sum.INPUT_UNCERTAINTY_Z,
   sum.ELEVATION_UOM,
   sum.ELEVATION_DATUM,
   sum.ELEVATION_DATUM_CONFIDENCE,
   sum.DEPTH_REFERENCE_TYPE,
   sum.DEPTH_REFERENCE_DATUM,
   sum.DEPTH_REFERENCE_HEIGHT,
   sum.DEPTH_REFERENCE_UOM,
   sum.alternate_names,
   sum.LOCATION_METHOD,
   sum.GDA94_LONGITUDE,
   sum.GDA94_LATITUDE,
   drt.DEPTH_REFERENCE_TYPE_ID


from

   borehole.boreholes b
   left join BOREHOLE.ALL_BOREHOLE_CURRENT_SUMMARY sum on b.borehole_id = sum.eno
   left join borehole.lu_depth_reference_types drt on sum.DEPTH_REFERENCE_TYPE = drt.TEXT


where
    drt.DEPTH_REFERENCE_TYPE_ID in ({})
and
    sum.ENO in ({})


""".format(','.join([str(x) for x in drt]),
           ','.join([str(x) for x in enos]))



df_header = pd.read_sql_query(header_query, con = ora_con)

# Now we do some processing

# Create a dataframe with only the important spatial information

df_header['Reference_datum_height_(mAGL)'] = np.nan
df_header['ground_elevation_(mAHD)'] = np.nan


# Iterating throught the rows
for index, row in df_header.iterrows():
    borehole_eno = row['ENO']

    # Get the ground elevation

    # Assert that all values are mAHD. We will deal with ellipsoids outside of the script

    if row['ELEVATION_DATUM'] != 'Australian Height Datum':

        elevation = np.nan

    else:

        elevation = row['ELEVATION_VALUE']

    if row['ELEVATION_UOM'] == 'mm':
        elevation = elevation / 1000.

    df_header.at[index, 'ground_elevation_(mAHD)'] = elevation

    # Now joint some of the key information to data frame

    height = row['DEPTH_REFERENCE_HEIGHT']

    # Do some checks of the uom
    if row['DEPTH_REFERENCE_UOM'] == 'mm':
        height = height / 1000.

    if row['DEPTH_REFERENCE_DATUM'] == "Australian height datum":

        if not np.isnan(elevation):
            height = height - elevation
        else:
            height = np.nan

    df_header.at[index, 'Reference_datum_height_(mAGL)'] = height


df_merged = df_master.merge(df_header, left_on = ['UWI', 'DEPTH_REFERENCE_ID'],
                            right_on = ['ENO', 'DEPTH_REFERENCE_TYPE_ID'])

# We add the more modern GDA2020 coordinates to the dataframe
transformer = Transformer.from_crs("EPSG:4283", "EPSG:7844", always_xy = True)

lons, lats = df_merged['GDA94_LONGITUDE'].values, df_merged['GDA94_LATITUDE'].values

df_merged['GDA2020_longitude'], df_merged['GDA2020_latitude'] =transformer.transform(lons, lats)

# Now we transform to projected coordinates. We infer the correct system from the longitude

# Create columns
df_merged['projected_crs'] = ''
df_merged['easting'] = np.nan
df_merged['northing'] = np.nan

transformer_z52 = Transformer.from_crs("EPSG:4283", "EPSG:28352", always_xy = True)
transformer_z53 = Transformer.from_crs("EPSG:4283", "EPSG:28353", always_xy = True)

z52_mask = (df_merged['GDA2020_longitude'] > 126.) & (df_merged['GDA2020_longitude'] < 132.)
z53_mask = (df_merged['GDA2020_longitude'] > 132.) & (df_merged['GDA2020_longitude'] < 138.)

# make sure we have covered all the coordinate reference systems
assert len(df_merged) == z52_mask.sum() + z53_mask.sum()

x_52, y_52 = transformer_z52.transform(df_merged['GDA94_LONGITUDE'], df_merged['GDA94_LATITUDE'])
x_53, y_53 = transformer_z53.transform(df_merged['GDA94_LONGITUDE'], df_merged['GDA94_LATITUDE'])

df_merged.at[z52_mask,'projected_crs'] = ["GDA94 / MGA zone 52"] * z52_mask.sum()
df_merged.at[z53_mask,'projected_crs'] = ["GDA94 / MGA zone 53"] * z53_mask.sum()

df_merged.at[z52_mask,'easting'] = x_52[z52_mask.values]
df_merged.at[z53_mask,'easting'] = x_53[z53_mask.values]

df_merged.at[z52_mask,'northing'] = y_52[z52_mask.values]
df_merged.at[z53_mask,'northing'] = y_53[z53_mask.values]


df_merged.to_csv("EFTF_induction_spatial_data.csv", index = False)
