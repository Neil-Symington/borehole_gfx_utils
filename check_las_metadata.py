# Script that checks if a las file metadata meets our minimum standard.
import yaml
import lasio
import numpy as np

# load the required fields from the yaml file
yaml_file = "lasfile_required_fields.yaml"
fields = yaml.safe_load(open(yaml_file))['metadata_fields']
# Define las file
#infile = "data/13BP01D_induction_gamma.LAS" #Fixed data file
infile = "data/13BP01D_induction_up.las" #'Raw' induction data


# Now extract the ones that have been assigned as essential as well as their data types
essential_fields = {}

for item in fields:
    if fields[item].get('required'):
        # Convert the data type to a numpy class
        if isinstance(fields[item]['dtype'], (list, tuple)):
            essential_fields[item] = tuple([getattr(np, d) for d in fields[item]['dtype']])
        elif isinstance(fields[item]['dtype'], (str)):
            essential_fields[item] = (getattr(np,fields[item]['dtype']))

# Open the las file

las = lasio.read(infile)

# we will flag the metadata as having passed for now. Innocent until proven guilty ;)
metadata_pass = True
missing_fields = []

# Iterate through each of the essential fields and assert that each field is present and of the correct data type
for item in essential_fields:
    try:
        header_item = getattr(las.well, item)
        # Check data types
        if not isinstance(header_item.value, (essential_fields[item])):
            print("Check data type for {}".format(item))
    except AttributeError:
        missing_fields.append(item)
        metadata_pass = False

if metadata_pass:
    print("File '{}' passed the metadata check".format(infile))
else:
    print("File '{}' fails the metadata check. The following fields were missing from the header: {}".format(infile,
                                                                                                              missing_fields))