# Script that populates columns and rows in the master spreadsheet using data within another spreadsheet extracted
# from the oracle rock properties database. This script assumes that the master spreadsheet already contains the borehole
# enos (or UWI), which is required for the oracle query.

# NOte that as data in oracle in in variable condition we don't recommend simply clicking run on  this script. the
# processing will need to be specific to the data

import pandas as pd
import lasio
import sys, os
sys.path.append("../functions")
from utilities import get_essential_fields, extract_step_from_las


# load the required fields from the yaml file
yaml_file = "lasfile_parsing_settings.yaml"

spatial_metadata_sheet = 'EFTF_induction_gamma_spatial_data.csv'

# load in the spatial metadata we extracted from oracle
df_master = pd.read_csv(spatial_metadata_sheet, keep_default_na=False)

# Now extract the ones that have been assigned as essential as well and other information

essential_fields = get_essential_fields(yaml_file)


# Create empty fields in our master spreadsheet

for item in essential_fields.keys():
    df_master[item + "_induction"] = ''
    df_master[item + "_gamma"] = ''

# Iterate though our master sheet and get the necessary field from the las files
for index, row in df_master.iterrows():
    # Iterate through techniques
    for s in ['induction', 'gamma']:
        # First we populate the fields that are already in the las files
        path = s + "_path"
        las_path = row[path]
        if os.path.isfile(las_path):
            las = lasio.read(las_path)
        else:
            las = None
            continue
            # Iterate through the essential fields and extract if they exist
        for item in essential_fields.keys():
            column = "_".join([item, s])
            try:
                header_item = getattr(las.well, item)
                value = header_item.value
                # For cases where the metadata value is an empty string
                if len(str(value)) > 0:
                    df_master.at[index, column] = value
                else:
                    pass

            except AttributeError:
                pass
        # Calculate the step from the log data since this metadata field is seemingly always wrong
        step = extract_step_from_las(las)
        column = "_".join(["STEP", s])
        df_master.at[index, column] = step


# Delete any empty columns

df_master.dropna(how='all', axis=1).to_csv('EFTF_induction_gamma_metadata_all.csv', index = False)

## TODO find a pythonic way of converting between between ellipsoid and geoid