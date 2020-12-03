# Python script for adding/ ammending metadata within las files based on inputs from a spreadsheet
# The spreadsheet is created using inputs from the oracle database as described in extract_las_metadata.py

import lasio
from lasio import HeaderItem
import pandas as pd
import numpy as np
import sys, os
sys.path.append("../functions")
from utilities import get_essential_fields, get_instrument_metadata

log = "induction"
#log = "gamma"

# load the required fields from the yaml file
yaml_file = "lasfile_parsing_settings.yaml"

instruments = get_instrument_metadata(yaml_file)

# Get the master spreadsheet
master_spreadsheet = 'EFTF_induction_gamma_metadata_all_corrected.csv'

df_master = pd.read_csv(master_spreadsheet, keep_default_na=False)

outdir = r'C:\temp\cleaned_las'

if not os.path.exists(outdir):
    os.mkdir(outdir)

# Now extract the ones that have been assigned as essential as well and other information

essential_fields = get_essential_fields(yaml_file, log)

for index, row in df_master.iterrows():
    path = "_".join([log, "path"])
    las_path = row[path]
    # Check that the file path exists as there are sum nulls
    if os.path.isfile(las_path):
        las = lasio.read(las_path)
    else:
        continue

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
    # Replace the channel metadata
    params = las.params.dictview().keys()
    for item in params:
        del las.params[item]
    # Add the instrument
    column = "_".join([log, "instrument"])
    instrument = instruments[row[column]]['instrument_name']
    las.params["instrument_name"] = lasio.HeaderItem('instrument_name', value=instrument)

    # Remove empty headers
    hdrs = las.well.dictview().keys()
    for item in hdrs:
        if str(las.well[item].value) == "":
            del las.well[item]

    outfile = os.path.join(outdir, "_".join([row['WELL'], log + ".las"]))
    las.write(outfile, version = 2.0)

