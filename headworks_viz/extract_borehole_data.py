'''
The purpose of this script is to extract borehole headworks and other construction data from the oracle database and
write it into a table.
Neil Symington
neil.symington@ga.gov.au
'''

import os
import pandas as pd
import numpy as np
import sys
sys.path.append(r'H:\scripts')
import connect_to_oracle
from pyproj import Transformer

infile = "EFTF_boreholes.csv"

df_master = pd.read_csv(infile)

# get the enos and names

mask = pd.notnull(df_master['eno'])

enos = df_master['eno'][mask].values.astype(int)

borehole_names = df_master['borehole_name'][~mask]

# connect to oracle database

# Set up a connection to the oracle database
ora_con = connect_to_oracle.connect_to_oracle()

header_query = """

select
   b.borehole_name,
   e.geom_original.sdo_point.x as orig_lon,
   e.geom_original.sdo_point.y as orig_lat,
   e.geom_original.sdo_point.z as orig_z,
   e.geom_original.sdo_srid as orig_srid,
   sum.ENO,
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
   sum.PREFERRED_TD_M,
   crs.SRID,
   crs.COORD_REF_SYS_NAME,
   drt.DEPTH_REFERENCE_TYPE_ID

from

   borehole.boreholes b
   left join BOREHOLE.ALL_BOREHOLE_CURRENT_SUMMARY sum on b.borehole_id = sum.eno
   left join A.ENTITIES e on b.borehole_id = e.eno
   left join sdo_coord_ref_sys crs on e.geom_original.sdo_srid = crs.SRID
   left join borehole.depth_reference_points drp on b.borehole_id = drp.borehole_id
   left join borehole.lu_depth_reference_types drt on sum.DEPTH_REFERENCE_TYPE = drt.TEXT

where

    sum.ENO in ({})
or
    b.borehole_name in ({})
or
    sum.alternate_names in ({})


""".format(','.join([str(x) for x in enos]),
           ','.join(["".join(["'", s, "'"]) for s in borehole_names]),
           ','.join(["".join(["'", s, "'"]) for s in borehole_names]))

df_header = pd.read_sql_query(header_query, con = ora_con)

# if the alternate name is the same as the name we get duplicates
df_header.drop_duplicates(inplace = True)

# We also want to tranform all bore into their projected coordinate system. As all bores are either zone 52 and 53
# we only need two tranformers.

transformers_from_GDA94 = {"GDA94 / MGA zone 50": Transformer.from_crs("EPSG:4283", "EPSG:28350", always_xy = True),
                           "GDA94 / MGA zone 51": Transformer.from_crs("EPSG:4283", "EPSG:28351", always_xy = True),
                           "GDA94 / MGA zone 52": Transformer.from_crs("EPSG:4283", "EPSG:28352", always_xy = True),
                           "GDA94 / MGA zone 53": Transformer.from_crs("EPSG:4283", "EPSG:28353", always_xy = True),
                           "GDA94 / MGA zone 54": Transformer.from_crs("EPSG:4283", "EPSG:28354", always_xy = True),
                           "GDA94 / MGA zone 55": Transformer.from_crs("EPSG:4283", "EPSG:28355", always_xy = True),
                           "GDA94 / MGA zone 56": Transformer.from_crs("EPSG:4283", "EPSG:28356", always_xy = True),
                           }

# Now we transform to projected coordinates. We infer the correct system from the longitude

# Create columns
df_header['projected_crs'] = ''
df_header['easting'] = np.nan
df_header['northing'] = np.nan
df_header['geographic_crs'] = "GDA94"
print(df_header)
# Iterate through the frame and make the transform
for index, row in df_header.iterrows():
    x, y = np.float64(row['GDA94_LONGITUDE']), np.float64(row['GDA94_LATITUDE'])
    if x < 120.:
        crs = "GDA94 / MGA zone 50"
    elif 120. < x < 126.:
        crs = "GDA94 / MGA zone 51"
    elif 126. < x <= 132.:
        crs = "GDA94 / MGA zone 52"
    elif 132. < x <= 138.:
        crs = "GDA94 / MGA zone 53"
    elif 138. < x < 144.:
        crs = "GDA94 / MGA zone 54"
    elif 144. < x <= 150.:
        crs = "GDA94 / MGA zone 55"
    elif 150. < x <= 156.:
        crs = "GDA94 / MGA zone 56"

    x_proj, y_proj = transformers_from_GDA94[crs].transform(x,y)
    df_header.at[index, 'easting'] = x_proj
    df_header.at[index, 'northing'] = y_proj
    df_header.at[index,'projected_crs'] = crs


df_header.to_csv("EFTF_header.csv", index = False)


# check that all the entries have ben recovered

for index, row in df_master.iterrows():
    eno = row['eno']
    if np.isnan(eno):
        if row['borehole_name'] not in df_header['BOREHOLE_NAME'].values and row['borehole_name'] not in df_header['ALTERNATE_NAMES'].values:
            print(row['borehole_name'], ' not a valid borename or alternate name')
    else:
        if int(eno) not in df_header['ENO'].values:
            print(str(int(eno)), ' is not a valid eno')


# Now we are going to get a table of construction information

construction_query = """

select
   b.borehole_name,
   b.borehole_id,
   drp.DEPTH_REFERENCE_ID,
   drp.depth_reference_type_id,
   bic.INTERVAL_COLLECTION_ID,
   bdi.INTERVALNO,
   bdi.interval_begin,
   bdi.interval_end,
   bcl.construction_name,
   blucl.*

from

   borehole.boreholes b
   left join borehole.depth_reference_points drp on b.borehole_id = drp.borehole_id
   left join borehole.interval_collections bic on drp.DEPTH_REFERENCE_ID = bic.DEPTH_REFERENCE_ID
   left join borehole.downhole_intervals bdi on bic.INTERVAL_COLLECTION_ID = bdi.INTERVAL_COLLECTION_ID
   right join borehole.construction_log bcl on bdi.INTERVALNO = bcl.INTERVALNO
   left join borehole.lu_construction_type blucl on bcl.construction_type_id = blucl.construction_type_id


where
    b.borehole_id in ({})


""".format(','.join([str(x) for x in df_header['ENO'].values]))

df_construction = pd.read_sql_query(construction_query, con = ora_con)

df_construction.to_csv('EFTF_construction.csv', index = False)
