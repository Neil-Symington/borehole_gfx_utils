# Access the borehole header information from oracle using a sql query

import sys
sys.path.append(r'H:\scripts')
##TODO add connect_to_oracle script to H:\ drive
import connect_to_oracle
import pandas as pd
# Connect to the database

ora_con = connect_to_oracle.connect_to_oracle()

# Create the sql query
enos= [621622,621623,621624,621625,626981,626984,626986,626987,
       626988,626989,626990,626991,626992,627061,627062,627063,
       627064,628981,635728,635729,635730,635732,635733,635734,
       635735,635736,635737,635738,635739,635740,635741,635742,
       635743,635744,635745,635746,635747,635748,635750,635921,
       635922,635923,635957,635958,635959,635960,636181,636182,
       636183,636184,636185,636186,636187,636189,636190,636191,
       636192,636193,636194,636195,636196,636197,636198,636200,
       636201,636203,636204]
st_eno = ','.join(str(x) for x in enos)

# Now bring in the borehole information from the database

header_query = """

select
   b.borehole_name,
   sum.ENO,
   sum.ELEVATION_VALUE,
   sum.INPUT_UNCERTAINTY_Z,
   sum.ELEVATION_UOM,
   sum.ELEVATION_DATUM,
   sum.ELEVATION_DATUM_CONFIDENCE,
   sum.DEPTH_REFERENCE_TYPE,
   sum.DEPTH_REFERENCE_DATUM,
   sum.DEPTH_REFERENCE_HEIGHT,
   sum.DEPTH_REFERENCE_UOM,
   sum.alternate_names,
   sum.LOCATION_METHOD,
   sum.ORIG_X_LONGITUDE,
   sum.INPUT_UNCERTAINTY_X,
   sum.ORIG_Y_LATITUDE,
   sum.INPUT_UNCERTAINTY_Y,
   sum.ORIG_INPUT_LOCATION_DATUM,
   drt.DEPTH_REFERENCE_TYPE_ID


from

   borehole.boreholes b
   left join BOREHOLE.ALL_BOREHOLE_CURRENT_SUMMARY sum on b.borehole_id = sum.eno
   left join borehole.lu_depth_reference_types drt on sum.DEPTH_REFERENCE_TYPE = drt.TEXT


where
    sum.ENO in ({})


""".format(st_eno)

print(header_query)

df_header = pd.read_sql(header_query, con=ora_con)