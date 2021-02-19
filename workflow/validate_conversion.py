# This script is used to check the similarity between the uncleaned and cleaned las files

import lasio
import pandas as pd
from scipy import stats
import numpy as np
import os, sys
sys.path.append(r"C:\Users\u77932\Documents\github\borehole_gfx_utils\functions")
from utilities import get_curve_metadata

root = r"C:\Users\u77932\Documents\github\borehole_gfx_utils\workflow"
infile = os.path.join(root, "EFTF_induction_gamma_metadata_all_corrected_.csv")

#log = "induction"
log = "gamma"

outfile = os.path.join(root, "".join(["EFTF_", log, "_validation.csv"]))

df_master = pd.read_csv(infile)

# load the required fields from the yaml file
yaml_file = os.path.join(root, "lasfile_parsing_settings.yaml")

curve_metadata = get_curve_metadata(yaml_file)

df_valid = df_master[[log + "_path", "WELL", "UWI"]].copy()

df_valid.dropna(subset = [log + "_path"], inplace = True)

clean_dir = os.path.join(r"\\prod.lan\active\proj\futurex\Common\Working\Neil\cleaned_las", log)

df_valid['clean_path'] = [os.path.join(clean_dir, "{}_{}.las".format(w, log)) for w in df_valid['WELL']]

# We will calculate the following stats
summ_stats  = ['nobs', 'mean', 'min', 'max', 'variance', 'nan_pos_errors']

# We will use the following columns
if log == "induction":
    curve_variables = ['DEPTH', 'INDUCTION_CALIBRATED', 'INDUCTION', 'DEEP_INDUCTION', 'MEDIUM_INDUCTION',
                       'IND_RES', 'DEEP_RES', 'MEDIUM_RES']
elif log == "gamma":
    curve_variables = ['DEPTH', 'GAMMA_CALIBRATED', 'GR']


# add the columns to to df_valid
for varname in curve_variables:
    for s in summ_stats:
        col = "_".join([varname, s])
        df_valid[col] = np.nan

def extract_statistics(arr1, arr2, varname):
    difference_array = np.absolute(np.subtract(arr1, arr2))
    summary = stats.describe(difference_array, nan_policy = 'omit')
    results = {}
    for s in summ_stats:
        key = "_".join([varname, s])
        if s == 'min':
            results[key] = np.nanmin(difference_array)
        elif s == 'max':
            results[key] = np.nanmax(difference_array)
        elif s=='nan_pos_errors':
            if not np.logical_and(np.any(np.isnan(arr1)), np.any(np.isnan(arr1))):
                results[key] = 0.
            elif np.all(np.where(np.isnan(arr1))[0] == np.where(np.isnan(arr2))[0]):
                results[key] = 0.
            else:
                results[key] = 1.
        else:
            results[key] = getattr(summary, s)
    return results

def dict2dataframe(dictionary, dataframe, ind):
    for key in dictionary:
        dataframe.at[ind, key] = dictionary[key]
    return(dataframe)


# Iterate through df_valid and check the curves

for index, row in df_valid.iterrows():
    if row['WELL'] == "RN017536" or row['WELL'] == "RN019449" or row["WELL"] == "RN019678":
        continue
    df_las1 = lasio.read(row[log + '_path']).df()
    df_las2 = lasio.read(row["clean_path"]).df()
    # run the depth check
    depth_results = extract_statistics(df_las1.index.values, df_las2.index.values, 'DEPTH')
    # add to the dataframe
    df_valid = dict2dataframe(depth_results, df_valid, index)
    # iterate through columns
    for var in df_las1.columns:
        if var not in curve_variables:
            continue
        arr1 = df_las1[var].values
        try:
            var2 = curve_metadata[var]['field_name']
        except KeyError:
            var2 = var
        arr2 = df_las2[var2].values
        var_results = extract_statistics(arr1, arr2, var)
        # add to dataframe
        df_valid = dict2dataframe(var_results, df_valid, index)
    df_valid.to_csv(r"C:\temp\validation_gamma.csv")