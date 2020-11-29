# Python script for adding/ ammending metadata within las files based on inputs from a spreadsheet
# The spreadsheet is created using inputs from the oracle database as described in get_metadata_from_oracle.py

import lasio
import pandas as pd
import os
import yaml

# load the required fields from the yaml file
yaml_file = "lasfile_required_fields.yaml"
settings = yaml.safe_load(open(yaml_file))['metadata_fields']

# Get the master spreadsheet
master_spreadsheet = 'EFTF_induction_spatial_data.csv'

df_master = pd.read_csv(master_spreadsheet, keep_default_na=False)

outdir = r'C:\temp\cleaned_las'

if not os.path.exists(outdir):
    os.mkdir(outdir)

required_fields = ['COMP', 'DATE', 'X', 'Y', 'GDAT', 'HZCS', 'LMF', 'APD', 'EPD', 'STRT', 'STOP', 'STEP', 'NULL']

### TODO figure out how to set attributes
for index, row in df_master.iterrows():
    file= row['prod_path']
    print(file)
    las = lasio.read(file)
    print(las.well)
    break
    for item in required_fields:
        try:
            fmt = settings[item]['format']
            setattr(las.well, item, fmt.format(row[item]))
        except KeyError:
            setattr(las.well, item, row[item])
    break




