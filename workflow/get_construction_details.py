import pandas as pd
import sys
sys.path.append(r'H:\scripts')
import connect_to_oracle


## TODO work this into the existing workflow

# Set up a connection to the oracle database
ora_con = connect_to_oracle.connect_to_oracle()

# Load the master spreadsheet

infile = "EFTF_induction_gamma_metadata_all_corrected.csv"
df_master = pd.read_csv(infile, keep_default_na=False)

enos = df_master['UWI'].values

# Create a sql query

header_query = """

select
   b.borehole_name,
   b.borehole_id,
   drp.DEPTH_REFERENCE_ID,
   drp.depth_reference_type_id,
   bic.INTERVAL_COLLECTION_ID,
   bdi.INTERVALNO,
   bdi.interval_begin,
   bdi.interval_end,
   bcl.construction_name,
   blucl.*
   
from

   borehole.boreholes b
   left join borehole.depth_reference_points drp on b.borehole_id = drp.borehole_id
   left join borehole.interval_collections bic on drp.DEPTH_REFERENCE_ID = bic.DEPTH_REFERENCE_ID
   left join borehole.downhole_intervals bdi on bic.INTERVAL_COLLECTION_ID = bdi.INTERVAL_COLLECTION_ID
   right join borehole.construction_log bcl on bdi.INTERVALNO = bcl.INTERVALNO
   left join borehole.lu_construction_type blucl on bcl.construction_type_id = blucl.construction_type_id


where
    b.borehole_id in ({})


""".format(','.join([str(x) for x in enos]))



df_header = pd.read_sql_query(header_query, con = ora_con)

df_header.to_csv(r'construction.csv')
