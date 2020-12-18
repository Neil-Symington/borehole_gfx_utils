
import numpy as np
import yaml
import scipy.stats as stats

def get_instrument_metadata(yaml_file):
    return yaml.safe_load(open(yaml_file))['instruments']

def get_essential_fields(yaml_file, log = None):
    # Load file
    fields = yaml.safe_load(open(yaml_file))['metadata_fields']
    # Create new dictionary
    essential_fields = {}
    for item in fields:
        if fields[item].get('required'):
            # Convert the data type to a numpy class
            if isinstance(fields[item]['dtype'], (list, tuple)):
                essential_fields[item] = {'dtype': tuple([getattr(np, d) for d in fields[item]['dtype']])}
            elif isinstance(fields[item]['dtype'], (str)):
                essential_fields[item] = {'dtype': (getattr(np, fields[item]['dtype']))}
            elif isinstance(fields[item]['dtype'], (float)):
                essential_fields[item] = {'dtype': (getattr(np, fields[item]['dtype']))}
            if "format" in fields[item].keys():
                essential_fields[item]['format'] = fields[item]["format"]
            if "use_col" in fields[item].keys():
                essential_fields[item]['use_col'] = fields[item]["use_col"]
            elif log is not None:
                essential_fields[item]['use_col'] = "_".join([item, log])
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
    return essential_fields

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