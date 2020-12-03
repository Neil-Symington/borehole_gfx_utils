# Written using Python 3.6
#
# Check a list of boreholes for concordance between collar elevations
# in A.ENTITIES and BOREHOLE.BOREHOLES databases.
#
# The list of boreholes is harvested from the names of folders in which raw
# gamma and induction conductivity field geophysics logs are stored.

import os, sys
import pandas as pd

# Connect to oracle using credentials stored in safe location
sys.path.append(r'H:\scripts')
import connect_to_oracle

oracon = connect_to_oracle.connect_to_oracle()
cur = oracon.cursor()

# Define key variables
data_folder = r"\\prod.lan\active\proj\futurex\StuartCorridor\Data\FieldworkAndPhotos\FieldData\Induction_and_Gamma"


# Functions
def get_borehole_list(folder):
    dirlist = [folder for folder in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, folder))]
    return dirlist


def get_entities_coords(borehole):
    """Get entity and coordinate information of a named borehole
    from the A.ENTITIES database in GA's Oracle environment
    Parameters:
    borehole (string):         The name of the borehole to query, eg RN018612
    Returns:
    A Pandas dataframe with columns:
    ENO, ENTITYID, ENTITY_TYPE, ACCESS_CODE, LON, LAT, Z, ORIG_LON, ORIG_LAT, ORIG_Z
    The dataframe will contain either a single row of data if the borehole is found,
    or will be empty.
    """

    # Construct SQL query
    query = '''
    select
        e.eno,
        e.entityid,
        e.entity_type,
        e.access_code,
        e.geom.sdo_point.x as lon,
        e.geom.sdo_point.y as lat,
        e.geom.sdo_point.z as z,
        e.geom.sdo_srid as srid,
        e.geom_original.sdo_point.x as orig_lon,
        e.geom_original.sdo_point.y as orig_lat,
        e.geom_original.sdo_point.z as orig_z,
        e.geom_original.sdo_srid as orig_srid
    from
        a.entities e
    where
        entityid in ('{}')
    '''.format(borehole)

    entities_coords = pd.read_sql(query, oracon)
    return entities_coords


def get_boreholes_coords(borehole):
    """Get entity and coordinate information of a named borehole
    from the BOREHOLE.BOREHOLES database in GA's Oracle environment
    Parameters:
    borehole (string):         The name of the borehole to query, eg RN018612
    Returns:
    A Pandas dataframe with columns:
    BOREHOLE_ID, BOREHOLE_NAME, LON, LAT, ELEVATION_VALUE, UOM, DATUM
    The dataframe will contain either a single row of data if the borehole is found,
    or will be empty.
    """

    # Construct SQL query
    query = '''
    select
        b.borehole_id,
        b.borehole_name,
        b.location.sdo_point.x as lon,
        b.location.sdo_point.y as lat,
        b.elevation_value,
        d.code as uom,
        ed.code as datum
    from
        borehole.boreholes b
        join borehole.lu_uom d
        on b.elevation_uom_id = d.uom_id
        join borehole.lu_elevation_datums ed
        on b.elevation_datum_id = ed.elevation_datum_id
    where
        borehole_name in ('{}')
    '''.format(borehole)

    borehole_coords = pd.read_sql(query, oracon)
    return borehole_coords


def get_datum_info(srid):
    query = '''
    select
        *
    from
        sdo_coord_ref_sys
    where
        srid = {}
    '''.format(srid)

    sdo_coord_ref_sys = pd.read_sql(query, oracon)
    return sdo_coord_ref_sys


# Mainline ---------------------------------------------------

dirlist = get_borehole_list(data_folder)
for folder in dirlist:

    # Get location information from A.ENTITIES table
    # ENO, ENTITYID, ENTITY_TYPE, ACCESS_CODE, LON, LAT, Z, SRID, ORIG_LON, ORIG_LAT, ORIG_Z, ORIG_SRID
    entities_info = get_entities_coords(folder)

    # Get location information from BOREHOLE.BOREHOLES table
    # BOREHOLE_ID, BOREHOLE_NAME, LON, LAT, ELEVATION_VALUE, UOM, DATUM
    boreholes_info = get_boreholes_coords(folder)

    # Reporting
    if boreholes_info.empty and entities_info.empty:
        print('{} not found in A.ENTITIES or BOREHOLE.BOREHOLES'.format(folder))
    elif entities_info.empty:
        print('{} not found in A.ENTITIES'.format(folder))
        print('Update A.ENTITIES.ENTITYID')
        print('\nBOREHOLE collar elevation stored as: {} {} {}'.format(
            boreholes_info.at[0, 'ELEVATION_VALUE'],
            boreholes_info.at[0, 'UOM'],
            boreholes_info.at[0, 'DATUM']
        ))
    elif boreholes_info.empty:
        print('{} not found in BOREHOLE.BOREHOLES'.format(folder))
        print('Update BOREHOLE.BOREHOLES.BOREHOLE_NAME')
        print('\nEntity elevation stored as: {} (GDA94 - 2D). No borehole elevation.'.format(
            entities_info.at[0, 'Z']
        ))
    else:
        orig_srid = entities_info.at[0, 'ORIG_SRID']
        orig_datum = get_datum_info(orig_srid).at[0, 'COORD_REF_SYS_NAME']
        entity_z = entities_info.at[0, 'Z']
        orig_z = entities_info.at[0, 'ORIG_Z']
        borehole_z = boreholes_info.at[0, 'ELEVATION_VALUE']
        borehole_uom = boreholes_info.at[0, 'UOM']
        borehole_datum = boreholes_info.at[0, 'DATUM']
        print('{} entity z: {} (GDA94 - 2D); orig entity z: {} {};     borehole z: {} {} {}'.format(
            folder,
            entities_info.at[0, 'Z'],
            entities_info.at[0, 'ORIG_Z'],
            orig_datum,
            boreholes_info.at[0, 'ELEVATION_VALUE'],
            boreholes_info.at[0, 'UOM'],
            boreholes_info.at[0, 'DATUM']
        ))