import pandas as pd

import numpy as np

def RawRec_to_RecAttr_fn(df_HumanRawRec, df_Human, cohort_args, record_args, attr_cols):
    #-------------------
    df = df_HumanRawRec

    raw_record_id = 'event_id' # <--------- 
    # filter out the records we don't need (optional) 
    df = df.drop_duplicates()
    df = df.groupby(raw_record_id).first().reset_index()

    # create a new column for raw record id (optional)
    df['patient_invitation_id_encoded'] = df['patient_id_encoded'] + ':' + df['invitation_id_encoded']

    # have a check that the raw record id is unique
    # print('have a check that the raw record id is unique:')
    # print(df[[raw_record_id]].value_counts().max() == 1)

    # update datetime columns
    df['event_date'] = pd.to_datetime(df['event_date'], format='mixed')
    df['invitation_date'] = pd.to_datetime(df['invitation_date'], format='mixed')

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
    # df = df.sort_values(['PID', 'PInvID', 'RxID', 'DT']).reset_index(drop = True)
    df = df.sort_values(prt_record_args['RecIDChain'] + ['DT']).reset_index(drop = True)
    # df.head()
    # print('Sort Records based on RootID and DT.')

    ParentID = record_args['prt_record_args']['RecID']
    df[RecID] = df[ParentID].astype(str) + '-' + df.groupby(ParentID).cumcount().apply(lambda x: str(x).zfill(3))
    # df.head()
    # print('Generated RecID for each record.')
    #-------------------

    df_HumanRecAttr = df[attr_cols].reset_index(drop = True)
    return df_HumanRecAttr


MetaDict = {
	"RawRec_to_RecAttr_fn": RawRec_to_RecAttr_fn
}