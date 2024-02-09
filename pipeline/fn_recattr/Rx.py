import pandas as pd

import numpy as np

def RawRec_to_RecAttr_fn(df_HumanRawRec, df_Human, cohort_args, record_args, fld_cols):
    #-------------------
    df = df_HumanRawRec

    # filter out the records we don't need (optional) 
    df = df.drop_duplicates()
    df = df.groupby('prescription_id_encoded').first().reset_index()

    # create a new column for raw record id (optional)
    df['patient_invitation_id_encoded'] = df['patient_id_encoded'] + ':' + df['invitation_id_encoded']

    # have a check that the raw record id is unique
    # print('have a check that the raw record id is unique:')
    # print(df[['prescription_id_encoded']].value_counts().max() == 1)

    # update datetime columns
    df['start_date'] = pd.to_datetime(df['start_date'], format='mixed')
    df['invitation_date'] = pd.to_datetime(df['invitation_date'], format='mixed')
    df['updated_date'] = pd.to_datetime(df['updated_date'], format='mixed')
    df['insurance_start_date'] = pd.to_datetime(df['insurance_start_date'], format='mixed')
    df['written_date'] = pd.to_datetime(df['written_date'], format='mixed')
    df['date_fdb_updated'] = pd.to_datetime(df['date_fdb_updated'], format='mixed')
            
    # select a DT. TODO: you might need to localize the datetime to local timezone. 
    df['DT'] = df['invitation_date']
    # print(df.shape)

    # merge with the parent record (a must except Human Records)
    df_Prt = record_args['df_Prt']
    prt_record_args = record_args['prt_record_args']
    df_merged = pd.merge(df_Prt, df, how = 'inner', on = prt_record_args['RawRecID'])
    # print(prt_record_args['RawRecID'])
    # print('df -->', df.shape, df.columns)
    # print('df_Prt -->', df_Prt.shape, df_Prt.columns)
    # print('df_merged -->', df_merged.shape, df_merged.columns)

    # sort the table by RootID and DT
    RecName = record_args['RecName']
    RecID = record_args['RecID']
    RootID = cohort_args['RootID']
    df = df_merged
    df = df.sort_values(['PID', 'PInvID', 'DT']).reset_index(drop = True)
    # df.head()
    # print('Sort Records based on RootID and DT.')

    # create a new column for RecID
    ParentID = record_args['prt_record_args']['RecID']
    df[RecID] = df[ParentID].astype(str) + '-' + df.groupby(ParentID).cumcount().apply(lambda x: str(x).zfill(3))
    # df.head()
    # print('Generated RecID for each record.')
    #-------------------

    df_HumanRecFld = df[fld_cols].reset_index(drop = True)
    return df_HumanRecFld


MetaDict = {
	"RawRec_to_RecAttr_fn": RawRec_to_RecAttr_fn
}