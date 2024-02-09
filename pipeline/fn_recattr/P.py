import pandas as pd

import numpy as np

def RawRec_to_RecAttr_fn(df_HumanRawRec, df_Human, cohort_args, record_args, attr_cols):
    #-------------------
    df = df_HumanRawRec

    # 1. filter out the records we don't need (optional) 
    df_filter = df_HumanRawRec.drop_duplicates()
    df = df_filter
    # 2. create a new column for raw record id (optional)

    # 3. update datetime columns 

    # 4. select a DT.

    # 5. merge with the parent record 
    df_Prt = record_args['df_Prt']
    prt_record_args = record_args['prt_record_args']
    df_merged = pd.merge(df_Prt, df, how = 'inner', on = prt_record_args['RawRecID'])
    # print(prt_record_args['RawRecID'])
    # print('df -->', df.shape, df.columns)
    # print('df_Prt -->', df_Prt.shape, df_Prt.columns)
    # print('df_merged -->', df_merged.shape, df_merged.columns)
    df = df_merged

    # 6. sort the table by RootID and DT
    RootID = cohort_args['RootID']
    df = df.sort_values(RootID).reset_index(drop = True)

    # 7. create a new column for RecID (no need)
    #-------------------

    df_HumanRecFld = df[attr_cols].reset_index(drop = True)
    return df_HumanRecFld


MetaDict = {
	"RawRec_to_RecAttr_fn": RawRec_to_RecAttr_fn
}