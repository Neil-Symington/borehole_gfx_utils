# Python script for adding/ ammending metadata within las files based on inputs from a spreadsheet
# The spreadsheet is created using inputs from the oracle database as described in merge_metadata_tables.py

import lasio
from lasio import SectionItems, HeaderItem
import pandas as pd
import numpy as np
import os
import yaml

# load the required fields from the yaml file
yaml_file = "lasfile_required_fields.yaml"
fields = yaml.safe_load(open(yaml_file))['metadata_fields']

# Get the master spreadsheet
master_spreadsheet = 'EFTF_induction_metadata_merged.csv'

df_master = pd.read_csv(master_spreadsheet, keep_default_na=False)

outdir = r'C:\temp\cleaned_las'

if not os.path.exists(outdir):
    os.mkdir(outdir)

# Now extract the ones that have been assigned as essential as well as their data types
essential_fields = {}

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
        if "use_col" in fields[item].keys():
            essential_fields[item]['use_col'] = fields[item]["use_col"]
        else:
            essential_fields[item]['use_col'] = item
        if "unit" in fields[item].keys():
            essential_fields[item]['unit'] = fields[item]["unit"]
        else:
            essential_fields[item]['unit'] = ""
        if "description" in fields[item].keys():
            essential_fields[item]['description'] = fields[item]["description"]
        else:
            essential_fields[item]['description'] = ""


for index, row in df_master.iterrows():
    file= row['prod_path']
    print(file)
    las = lasio.read(file)
    for item in essential_fields:
        col = essential_fields[item]["use_col"]
        value = row[col]
        ## This is a bit of a hack as only floats have formats currently
        if "format" in essential_fields[item]:
            value = np.float(value)
            fmt = essential_fields[item]['format']
            value = fmt.format(value)
        # If the fields exist already, the values can be assigned using setattr()
        if item in las.well.keys():
            setattr(las.well, item, value)
        else:
            new_header = HeaderItem(mnemonic = item,
                                unit= essential_fields[item]["unit"],
                                value = value,
                                descr = essential_fields[item]["description"])
            las.well[item] = new_header
    # Remove empty headers
    for item in las.well:
        if str(item.value) == "":
            del las.well[item.mnemonic]

    outfile = os.path.join(outdir, row['WELL'] + "_induction.las")
    las.write(outfile, version = 2.0)
