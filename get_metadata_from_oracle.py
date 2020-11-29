# Script that populates columns and rows in the master spreadsheet using data within another spreadsheet extracted
# from the oracle rock properties database. This script assumes that the master spreadsheet already contains the borehole
# enos (or UWI), which is required for the oracle query.

# NOte that as data in oracle in in variable condition we don't recommend simply clicking run on  this script. the
# processing will need to be specific to the data

import pandas as pd
import lasio
import numpy as np
import yaml

# load the required fields from the yaml file
yaml_file = "lasfile_required_fields.yaml"
fields = yaml.safe_load(open(yaml_file))['metadata_fields']

spatial_metadata_sheet = 'EFTF_induction_spatial_data.csv'

df_or = pd.read_csv(oracle_sheet)

master_spreadsheet = 'EK_induction_master.csv'

df_master = pd.read_csv(master_spreadsheet, keep_default_na=False)

# Now extract the ones that have been assigned as essential as well as their data types
essential_fields = {}

keys = ['COMP', 'DATE', 'X', 'Y', 'GDAT', 'HZCS', 'LMF', 'APD', 'STRT', 'STOP', "NULL"]

def extract_step_from_las(las):
    """

    :param las: object
        las class from lasio
    :return:
        the depth interval or STEP
    """
    intervals = las['DEPT'][1:] - las['DEPT'][:-1]

    # due to floating point errors we will round it to 3 decimal places and find the uniqe value
    return np.unique(np.round(intervals, 3))[0]


for item in keys:
    # Convert the data type to a numpy class
    if isinstance(fields[item]['dtype'], (list, tuple)):
        essential_fields[item] = tuple([getattr(np, d) for d in fields[item]['dtype']])
    elif isinstance(fields[item]['dtype'], (str)):
        essential_fields[item] = (getattr(np,fields[item]['dtype']))

# Iterate though our master sheet and get the necessary field from the las file
for index, row in df_master.iterrows():
    print(row['WELL'])
    # First we populate the fields that are already in the
    las_path = row['local_path']
    las = lasio.read(las_path)
    # Extract the step since this is always wrong
    step = extract_step_from_las(las)
    df_master.at[index, 'STEP'] = step

    for item in keys:
        try:
            header_item = getattr(las.well, item)
            value = header_item.value
            # Check data types
            if not isinstance(value, (essential_fields[item])):
                print("Check data type for {}".format(item))
            # For cases where the metadata value is an empty string
            if len(str(value)) > 0:
                df_master.at[index, item] = value
            else:
                pass


        except AttributeError:
            pass


# Get the necessary field from the oracle spreadsheet. Since the db is our 'point of truth' we will over write
# any of the x, y, z fields that are in the las file header

df_merged = df_master.merge(df_or, left_on = ['UWI', 'DEPTH_REFERENCE_ID'],
                            right_on = ['ENO', 'DEPTH_REFERENCE_TYPE_ID'])

# Unless there are nulls we will replace the X values in the master spreadsheet
xmask = pd.notnull(df_merged['ORIG_X_LONGITUDE'])
df_master.at[xmask, 'X'] = df_merged[xmask]['ORIG_X_LONGITUDE'].values

ymask = pd.notnull(df_merged['ORIG_Y_LATITUDE'])
df_master.at[ymask, 'Y'] = df_merged[ymask]['ORIG_Y_LATITUDE'].values

gdat_mask = pd.notnull(df_merged["ELEVATION_DATUM"])
df_master.at[gdat_mask, 'GDAT'] = df_merged[gdat_mask]["ELEVATION_DATUM"].values

hzcs_mask = pd.notnull(df_merged["ORIG_INPUT_LOCATION_DATUM"])
df_master.at[hzcs_mask, 'HZCS'] = df_merged[hzcs_mask]['ORIG_INPUT_LOCATION_DATUM'].values

# Now add the depth reference information

dr_mask = pd.notnull(df_merged['DEPTH_REFERENCE_HEIGHT'])
# Do some corrections on these heights to convert all to metres above ground

# get the unit of measurements
drd_uom = df_merged["DEPTH_REFERENCE_UOM"]
mm_mask = drd_uom == 'mm'
# drop column to reduce the risk of error
df_merged.drop(columns = "DEPTH_REFERENCE_UOM", inplace = True)

# Use the mask to convert to m
df_merged.at[mm_mask, "DEPTH_REFERENCE_HEIGHT"] = df_merged[mm_mask]['DEPTH_REFERENCE_HEIGHT'] * np.power(10,-3.)

# Convert elevation to mAHD
elev_uom = df_merged["ELEVATION_UOM"]
elev_mm_mask = elev_uom == 'mm'
# drop column to reduce the risk of error
df_merged.drop(columns = "ELEVATION_UOM",inplace = True)

df_merged.at[elev_mm_mask, "ELEVATION_VALUE"] = df_merged["ELEVATION_VALUE"] *  np.power(10,-3.)

# Find where the depth reference is relative to the geoid and subtract
depth_ref_datum = df_merged['DEPTH_REFERENCE_DATUM']
ahd_mask = depth_ref_datum == "Australian height datum"

df_merged.at[ahd_mask, "DEPTH_REFERENCE_HEIGHT"] = df_merged[ahd_mask]['DEPTH_REFERENCE_HEIGHT'] - df_merged["ELEVATION_VALUE"][ahd_mask]
# drop column to reduce the risk of error
df_merged.drop(columns = "DEPTH_REFERENCE_DATUM",inplace = True)

# Add to master spreadsheet

df_master.at[dr_mask, 'LMF'] = df_merged[dr_mask]['DEPTH_REFERENCE_TYPE'] # reference from
df_master.at[dr_mask, 'APD'] = df_merged[dr_mask]['DEPTH_REFERENCE_HEIGHT'] # reference height

df_master.at[dr_mask, 'EPD'] = df_merged[dr_mask]['ELEVATION_VALUE']

df_master.to_csv('EK_induction_master_metadata.csv')
