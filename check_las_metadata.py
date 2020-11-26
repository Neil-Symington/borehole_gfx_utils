# Script that checks if a las file metadata meets our minimum standard.
import yaml
import lasio
import numpy as np
import pandas as pd

# load the required fields from the yaml file
yaml_file = "lasfile_required_fields.yaml"
fields = yaml.safe_load(open(yaml_file))['metadata_fields']
# Define las file

master_spreadsheet = 'EK_induction_master.csv'
df_master = pd.read_csv(master_spreadsheet)

las_file_paths = df_master['local_path'].values

#las_file_paths = ["data/13BP01D_induction_gamma.LAS", "data/13BP01D_induction_up.las"]

# Define file path that the results will be written to
outfile = "EK_metadata_check_results.csv"

# Now extract the ones that have been assigned as essential as well as their data types
essential_fields = {}

for item in fields:
    if fields[item].get('required'):
        # Convert the data type to a numpy class
        if isinstance(fields[item]['dtype'], (list, tuple)):
            essential_fields[item] = tuple([getattr(np, d) for d in fields[item]['dtype']])
        elif isinstance(fields[item]['dtype'], (str)):
            essential_fields[item] = (getattr(np,fields[item]['dtype']))

# Function for checking metadata fields

def check_metadata(infile, essential_fields):
    """
    Function for checking the metadata of a las file
    :param infile: string
        .las file path
    :param essential_fields: dictionary
         dictionoary with essential las metadata fields as the keys and the expected data types as entries
    :param outfile: str
        path for the results of the metadata check to be written. If None the results are printed to screen.
    :return: dictionary
        dictionary with missing metadata fields and fields where the datatype needs checking

    """
    # Open the las file
    print("\nOpening ", infile)

    las = lasio.read(infile)

    # we will flag the metadata as having passed for now. Innocent until proven guilty ;)
    metadata_pass = True
    missing_fields = []
    datatype_mismatch_fields = []

    # Iterate through each of the essential fields and assert that each field is present and of the correct data type
    for item in essential_fields:
        try:
            header_item = getattr(las.well, item)
            # Check data types
            if not isinstance(header_item.value, (essential_fields[item])):
                print("Check data type for {}".format(item))
                datatype_mismatch_fields.append(item)
            # For cases where the metadata value is an empty string
            if len(str(header_item.value)) == 0:
                missing_fields.append(item)
                metadata_pass = False
        except AttributeError:
            missing_fields.append(item)
            metadata_pass = False

    if metadata_pass:
        print("File '{}' passed the metadata check".format(infile))
    else:
        print("File '{}' fails the metadata check. The following fields were missing from the header: {}".format(infile,
                                                                                                                 missing_fields))
    return {'passed_check': metadata_pass, 'missing_fields': missing_fields,
            'datatype_mismatch_fields': datatype_mismatch_fields}

results = {}

# Iterate through the lasfile paths
for file in las_file_paths:
    results[file] = check_metadata(file, essential_fields)

df = pd.DataFrame(results).T

df.to_csv(outfile, index_label = "lasFile")