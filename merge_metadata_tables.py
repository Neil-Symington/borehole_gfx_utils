# Script that populates columns and rows in the master spreadsheet using data within another spreadsheet extracted
# from the oracle rock properties database. This script assumes that the master spreadsheet already contains the borehole
# enos (or UWI), which is required for the oracle query.

# NOte that as data in oracle in in variable condition we don't recommend simply clicking run on  this script. the
# processing will need to be specific to the data

import pandas as pd
import lasio
import numpy as np
import yaml
import scipy.stats as stats

def extract_step_from_las(las):
    """

    :param las: object
        las class from lasio
    :return:
        the depth interval or STEP
    """
    intervals = las['DEPT'][1:] - las['DEPT'][:-1]

    # due to floating point errors we will round it to 3 decimal places and find the uniqe value
    return stats.mode(np.round(intervals, 3))[0][0]


# load the required fields from the yaml file
yaml_file = "lasfile_required_fields.yaml"
fields = yaml.safe_load(open(yaml_file))['metadata_fields']

spatial_metadata_sheet = 'EFTF_induction_spatial_data.csv'

df_or = pd.read_csv(spatial_metadata_sheet)

master_spreadsheet = 'EFTF_induction_master.csv'

df_master = pd.read_csv(master_spreadsheet, keep_default_na=False)

# Now extract the ones that have been assigned as essential as well as their data types
essential_fields = {}

keys = ['COMP', 'DATE', 'X', 'Y', 'GDAT', 'HZCS', 'LMF', 'APD', 'STRT', 'STOP', "NULL"]

for item in fields:
    if fields[item].get('required'):
        # Convert the data type to a numpy class
        if isinstance(fields[item]['dtype'], (list, tuple)):
            essential_fields[item] = {'dtype': tuple([getattr(np, d) for d in fields[item]['dtype']])}
        elif isinstance(fields[item]['dtype'], (str)):
            essential_fields[item] = {'dtype': (getattr(np,fields[item]['dtype']))}
        elif isinstance(fields[item]['dtype'], (float)):
            essential_fields[item] = {'dtype':(getattr(np, fields[item]['dtype']))}
        if "format" in fields[item].keys():
            essential_fields[item]['format'] = fields[item]["format"]


# Iterate though our master sheet and get the necessary field from the las file
for index, row in df_master.iterrows():
    print(row['WELL'])
    # First we populate the fields that are already in the
    las_path = row['prod_path']
    las = lasio.read(las_path)

    for item in essential_fields.keys():
        try:
            header_item = getattr(las.well, item)
            value = header_item.value
            # Check data types
            if not isinstance(value, (essential_fields[item]['dtype'])):
                print("Check data type for {}".format(item))
            # For cases where the metadata value is an empty string
            if len(str(value)) > 0:
                df_master.at[index, item] = value
            else:
                pass

        except AttributeError:
            pass
    # Calculate the step since this is always wrong
    step = extract_step_from_las(las)
    df_master.at[index, 'STEP'] = step


# Get the necessary field from the oracle spreadsheet. It is worth keeping all metadata at this stage for comparison

df_merged = df_master.merge(df_or, left_on = ['UWI', 'DEPTH_REFERENCE_ID'],
                                   right_on = ['UWI', 'DEPTH_REFERENCE_TYPE_ID'],
                            suffixes = [None, "_oracle"])

df_merged.to_csv('EFTF_induction_metadata_merged.csv', index = False)
