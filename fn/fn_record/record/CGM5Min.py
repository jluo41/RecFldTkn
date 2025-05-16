import pandas as pd

import numpy as np

OneRecord_Args = {'RecordName': 'CGM5Min',
 'RecID': 'CGM5MinID',
 'RecIDChain': ['PID'],
 'RawHumanID': 'PatientID',
 'ParentRecName': 'P',
 'RecDT': 'DT_s',
 'RawNameList': ['ElogBGEntry'],
 'human_group_size': 100,
 'rec_chunk_size': 100000}

RawName_to_RawConfig = {'ElogBGEntry': {'raw_columns': ['BGEntryID', 'PatientID', 'ObservationDateTime', 'BGValue',
                                 'ObservationEntryDateTime', 'TimezoneOffset',
                                 'EntryCreatedDateTime', 'ActualBGValue', 'ExternalSourceID',
                                 'UserObservationDateTime'],
                 'raw_base_columns': ['BGEntryID', 'PatientID', 'ObservationDateTime',
                                      'ObservationEntryDateTime', 'TimezoneOffset',
                                      'EntryCreatedDateTime', 'ExternalSourceID',
                                      'UserObservationDateTime'],
                 'rec_chunk_size': 100000,
                 'raw_datetime_column': 'ObservationDateTime'}}

attr_cols = ['PID', 'PatientID', 'CGM5MinID', 'DT_s', 'BGValue']

def get_RawRecProc_for_HumanGroup(df_RawRec_for_HumanGroup, OneRecord_Args, df_Human):
    df = df_RawRec_for_HumanGroup

    # 1. filter out the records we don't need (optional) 

    ##########
    # index = (df['ExternalSourceID'] == 18) # | (df['MeterType'] == 5)  # CGM
    # df = df[index].reset_index(drop = True)
    ##########



    df = df[df['TimezoneOffset'].abs() < 1000].reset_index(drop = True)

    # 2. create a new column for raw record id (optional)

    # 3. update datetime columns 
    DTCol_list = [
        'ObservationDateTime', 
        # 'ParentEntryID', 'ActivityTypeID',
        # 'ObservationEntryDateTime', 
        # 'EntryCreatedDateTime',
        # 'UserObservationDateTime'
        ]

    for DTCol in DTCol_list: 
        df[DTCol] = pd.to_datetime(df[DTCol], format = 'mixed')

    # x1. localize the datetime columns to based on time zone. 
    a = len(df)
    df = pd.merge(df, df_Human[['PatientID', 'user_tz']],  how = 'left')
    b = len(df)
    assert a == b

    # Ensure 'TimezoneOffset' is of the correct type and handle missing values
    # df['DT_tz'] = (
    #     df['TimezoneOffset']
    #     .infer_objects(copy=False)
    #     .replace(0, pd.NA)  # Use pd.NA for missing values
    #     .fillna(df['user_tz'])  # Fill missing values with 'user_tz'
    #     .astype('Int64')  # Ensure the column is of integer type, if applicable
    # )

    # df['DT_tz'] = (
    #     df['TimezoneOffset']
    #     .infer_objects(copy=False)
    #     .replace(0, pd.NA)  # Use pd.NA for missing values
    #     .fillna(df['user_tz'])  # Fill missing values with 'user_tz'
    #     .infer_objects(copy=False)  # Infer the data type after fillna
    #     .astype('Int64')  # Ensure the column is of integer type, if applicable
    # )

    df['DT_tz'] = pd.Series(
        df['TimezoneOffset'].replace(0, pd.NA).to_numpy(),
        dtype=pd.Int64Dtype()
    ).fillna(df['user_tz'])



    # print(df['DT_tz'])

    # If 'DT_tz' is supposed to be a datetime column, convert it
    # df['DT_tz'] = pd.to_datetime(df['DT_tz'], errors='coerce')
    # df['DT_tz'] = df['TimezoneOffset'].infer_objects(copy=False).replace(0, None).fillna(df['user_tz'])


    # DTCol = 'DT_r'
    # DTCol_source = 'EntryCreatedDateTime'
    # df[DTCol] = df[DTCol_source]
    # df[DTCol] = pd.to_datetime(df[DTCol]) + pd.to_timedelta(df['DT_tz'], 'm')
    # assert df[DTCol].isna().sum() == 0


    DTCol = 'DT_s'
    DTCol_source = 'ObservationDateTime'
    df[DTCol] = df[DTCol_source]
    df[DTCol] = pd.to_datetime(df[DTCol]).apply(lambda x: None if x <= pd.to_datetime('2010-01-01') else x)


    df[DTCol] = pd.to_datetime(df[DTCol]) + pd.to_timedelta(df['DT_tz'], 'm')


    # df[DTCol] = df[DTCol].fillna(df['DT_r'])


    df = df[df[DTCol].notna()].reset_index(drop = True)
    # assert df[DTCol].isna().sum() == 0

    DTCol_list = ['DT_s', 
                  # 'DT_r'
                  ] # 'DT_e'
    for DTCol in DTCol_list:
        # DateTimeUnit ='5Min'
        date = df[DTCol].dt.date.astype(str)
        hour = df[DTCol].dt.hour.astype(str)
        minutes = ((df[DTCol].dt.minute / 5).astype(int) * 5).astype(str)
        df[DTCol] = pd.to_datetime(date + ' ' + hour +':' + minutes + ':' + '00')

    # x3. drop duplicates
    df = df.drop_duplicates()

    # 4. select a DT as the RecDT
    RecDT = 'DT_s'

    # ----------------------------------------------------------------- #
    # x4. get the BGValue mean by RecDT (5Min)
    RawHumanID = OneRecord_Args['RawHumanID']
    df = df.groupby([RawHumanID, RecDT])[['BGValue']].mean().reset_index()
    # ----------------------------------------------------------------- #

    df_RawRecProc = df
    return df_RawRecProc 


MetaDict = {
	"OneRecord_Args": OneRecord_Args,
	"RawName_to_RawConfig": RawName_to_RawConfig,
	"attr_cols": attr_cols,
	"get_RawRecProc_for_HumanGroup": get_RawRecProc_for_HumanGroup
}