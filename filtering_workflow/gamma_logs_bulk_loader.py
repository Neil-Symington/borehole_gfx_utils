# For the oracle database we will adjust all observations to depth below ground level
import os, glob
import pandas as pd

# Get master spreadsheet
df_master = pd.read_csv(r"C:\Users\u77932\Documents\github\borehole_gfx_utils\workflow\EFTF_induction_gamma_metadata_all_corrected.csv")

outdir = r"\\prod.lan\active\proj\futurex\Common\Working\Neil\filtered_borehoel_gfx\gamma\data"

# We will write all of our data into a spreadsheet that approximates the bulk loader
df_bulk_loader = pd.DataFrame(columns = ['BOREHOLE_ID', 'PROCESSNO', 'PROPERTY', 'DEPTH', 'VALUE', 'UOM',
                                         'RESULT_QUALIFIER', 'ORIGINATOR',
                                         'UNCERTAINTY TYPE', 'SOURCETYPENO', 'SOURCE', 'NUMERICALCONFIDENCENO',
                                         'METADATAQUALITYNO', 'SUMMARYCONFIDENCENO', 'INSTRUMENT'])

data_dir = r'\\prod.lan\active\proj\futurex\Common\Working\Neil\filtered_borehoel_gfx\gamma\data'

var = 'GAMMA_CALIBRATED'
uom = 'cps'


for file in glob.glob(os.path.join(data_dir, '*_GAMMA_*')):
    print(file)
    path, fname = os.path.split(file)
    well = fname.split("_GAMMA")[0]
    df_well = pd.read_csv(file, index_col = 0)
    # now get the metadata row
    row = df_master[df_master['WELL'] == well]
    print(well)
    try:
        ref = row['Reference_datum_height_(mAGL)'].values[0]
    except IndexError:
        print(well)
        ref = 0
    # subtract this value from the depths and create a new column in df_well
    df_well['DEPTH'] = df_well.index.values - ref
    # Now add the eno
    df_well['BOREHOLE_ID'] = row['ENO'].values[0]
    df_well.rename(columns = {'filtered': 'VALUE'}, inplace = True)
    # add instrument
    instrument = row['gamma_instrument'].values[0]
    df_well['INSTRUMENT'] = instrument
    df_bulk_loader = df_bulk_loader.append(df_well)


df_bulk_loader['PROPERTY'] = 'natural gamma'
df_bulk_loader['UOM'] = uom
df_bulk_loader['RESULT_QUALIFIER'] = 3
df_bulk_loader['ORIGINATOR'] = 'Tan, K. '
df_bulk_loader['UNCERTAINTY TYPE'] = 'unknown'
df_bulk_loader['SOURCETYPENO'] = 3
df_bulk_loader['SOURCE'] = outdir
df_bulk_loader['NUMERICALCONFIDENCENO'] = 3
df_bulk_loader['METADATAQUALITYNO'] = 3
df_bulk_loader['SUMMARYCONFIDENCENO'] = 5

df_bulk_loader.reset_index().to_csv('EFTF1_' + var + '_bulk_loader.csv', index = False)