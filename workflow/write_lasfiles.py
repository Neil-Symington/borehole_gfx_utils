# Python script for adding/ ammending metadata within las files based on inputs from a spreadsheet
# The spreadsheet is created using inputs from the oracle database as described in extract_las_metadata.py

import lasio
from lasio import HeaderItem
import pandas as pd
import numpy as np
import sys, os
sys.path.append("../functions")
from utilities import get_essential_fields, get_instrument_metadata, get_curve_metadata

#log = "induction"
log = "gamma"

# load the required fields from the yaml file
yaml_file = "lasfile_parsing_settings.yaml"

instruments = get_instrument_metadata(yaml_file)
curve_metadata = get_curve_metadata(yaml_file)

# Get the master spreadsheet
master_spreadsheet = 'EFTF_induction_gamma_metadata_all_corrected.csv'

df_master = pd.read_csv(master_spreadsheet, keep_default_na=False)

outdir = os.path.join(r'\\prod.lan\active\proj\futurex\Common\Working\Neil\cleaned_las', log)

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
        if not essential_fields[item]['required']:
            del las.well[item]
            continue
        col = essential_fields[item]["use_col"]
        value = row[col]
        if "format" in essential_fields[item]:
            value = np.float(value)
            fmt = essential_fields[item]['format']
            value = fmt.format(value)

        # If the fields exist already, the values can be reassigned using setattr()
        if item in las.well.keys():
            setattr(las.well, item, value)
            # Make the description upper case if it is not already
            las.well[item].descr = las.well[item].descr.upper()
        else:
            new_header = HeaderItem(mnemonic = item,
                                    unit= essential_fields[item]["unit"].upper(),
                                    value = value,
                                    descr = essential_fields[item]["description"].upper())
            las.well[item] = new_header



    # Replace the channel metadata
    params = las.params.dictview().keys()
    for item in params:
        del las.params[item]
    # Add the instrument
    column = "_".join([log, "instrument"])
    instrument = instruments[row[column]]['instrument_name']
    las.params["INSTRUMENT_NAME"] = lasio.HeaderItem('INSTRUMENT_NAME', value=instrument)
    if log == 'induction':
        channel_offset = instruments[row[column]]['channel_offset']
        if len(channel_offset) == 1:
            las.params["intercoil_spacing"] = lasio.HeaderItem('INTERCOIL_SPACING', value=channel_offset[0], unit = 'M')
        else:
            las.params["med_intercoil_spacing"] = lasio.HeaderItem('MEDIUM_CHANNEL_INTERCOIL_SPACING',
                                                                    value=channel_offset[0], unit='M')
            las.params["deep_intercoil_spacing"] = lasio.HeaderItem('DEEP_CHANNEL_INTERCOIL_SPACING',
                                                                    value=channel_offset[1], unit='M')


    # Remove empty headers
    hdrs = las.well.dictview().keys()
    for item in hdrs:
        if str(las.well[item].value) == "":
            del las.well[item]

    # Add some miscellaneous metadata to the 'other' block
    las.other += '\n'
    las.other += "Converted from the raw data file: " + las_path
    las.other += "\n"
    las.other += "This data was collected by a sonde descending towards the bottom of the borehole from the surface."
    if len(row['ALTERNATE_NAMES']) > 0:
        las.other += "\n"
        las.other += "Alternative well names: " + ','.join(row['ALTERNATE_NAMES'].split(','))

    # Here we ensure we keep the precision from the original files
    df = las.df()
    column_fmt = {0: "%.3f"}
    for i, item in enumerate(df.columns):
        str_curve = list(df[item].astype(str).values)
        sig_fig = max([len(s.split('.')[-1]) for s in str_curve])
        fmt = "%.{}f".format(sig_fig)
        column_fmt[i+1] = fmt

    # Finally we rename some of the fields
    for field in las.curves[:]:
        field_name = field.mnemonic
        if field_name in curve_metadata.keys():
            if curve_metadata[field_name]['delete']:
                print(las_path)
                del las.curves[field_name]
                continue
            else:
                new_field = curve_metadata[field_name]['field_name']
                las.curves[field_name].mnemonic  = new_field
                las.curves[new_field].unit = curve_metadata[field_name]['unit'].upper()
                las.curves[new_field].descr = curve_metadata[field_name]['descr'].upper()
        else:
            pass

    outfile = os.path.join(outdir, "_".join([row['WELL'], log + ".las"]))
    las.write(outfile, version = 2.0, STEP = np.round(float(row['STEP_' + log]),3), column_fmt =column_fmt,
              wrap=False)
