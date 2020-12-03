import os
import pandas as pd
import numpy as np
import sys
sys.path.append(r'H:\scripts')
import connect_to_oracle
from pyproj import Transformer


# Set up a connection to the oracle database
ora_con = connect_to_oracle.connect_to_oracle()

# Load the master spreadsheet

infile = "EFTF_induction_gamma_master.csv"
df_master = pd.read_csv(infile, keep_default_na=False)

enos = df_master['UWI'].values
drt = np.unique(df_master['DEPTH_REFERENCE_ID'].values)

# Create a sql query

header_query = """

select
   b.borehole_name,
   e.geom_original.sdo_point.x as orig_lon,
   e.geom_original.sdo_point.y as orig_lat,
   e.geom_original.sdo_point.z as orig_z,
   e.geom_original.sdo_srid as orig_srid,
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
   crs.SRID,
   crs.COORD_REF_SYS_NAME,
   drt.DEPTH_REFERENCE_TYPE_ID


from

   borehole.boreholes b
   left join BOREHOLE.ALL_BOREHOLE_CURRENT_SUMMARY sum on b.borehole_id = sum.eno
   left join A.ENTITIES e on b.borehole_id = e.eno
   left join sdo_coord_ref_sys crs on e.geom_original.sdo_srid = crs.SRID
   left join borehole.lu_depth_reference_types drt on sum.DEPTH_REFERENCE_TYPE = drt.TEXT


where
    drt.DEPTH_REFERENCE_TYPE_ID in ({})
and
    sum.ENO in ({})


""".format(','.join([str(x) for x in drt]),
           ','.join([str(x) for x in enos]))


df_header = pd.read_sql_query(header_query, con = ora_con)

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


# We also want to tranform all bore into their projected coordinate system. As all bores are either zone 52 and 53
# we only need two tranformers.

transformers_from_GDA94 = {"GDA94 / MGA zone 52": Transformer.from_crs("EPSG:4283", "EPSG:28352", always_xy = True),
                           "GDA94 / MGA zone 53": Transformer.from_crs("EPSG:4283", "EPSG:28353", always_xy = True)}

# Now we transform to projected coordinates. We infer the correct system from the longitude

# Create columns
df_merged['projected_crs'] = ''
df_merged['easting'] = np.nan
df_merged['northing'] = np.nan
df_merged['geographic_crs'] = "GDA94"
# Iterate through the frame and make the transform
for index, row in df_merged.iterrows():
    x, y = np.float64(row['GDA94_LONGITUDE']), np.float64(row['GDA94_LATITUDE'])
    if 126. < x <= 132.:
        crs = "GDA94 / MGA zone 52"
    elif 132. < y <= 136.:
        crs = "GDA94 / MGA zone 53"
    x_proj, y_proj = transformers_from_GDA94[crs].transform(x,y)
    df_merged.at[index, ['easting', 'northing']] = [x_proj, y_proj]
    df_merged.at[index,'projected_crs'] = crs

df_merged.to_csv("EFTF_induction_gamma_spatial_data.csv", index = False)
