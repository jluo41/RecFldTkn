import os

import pandas as pd

import numpy as np

OneCohort_Args = {'CohortLabel': 2,
 'CohortName': '20240503_Survey',
 'FolderPath': '../_Data/0-Data_Raw/20240503_Survey/',
 'SourcePath': '../_Data/0-Data_Raw/20240503_Survey/Source',
 'Source2CohortName': 'ParquetV240730'}

SourceFile_SuffixList = ['parquet']

def get_RawName_from_SourceFile(file_path, OneCohort_Args):
    # RawName = file_path.split('_df_')[0].split('/')[-1] # split('.')[0]
    RawName = file_path.split('_df')[0].split('/')[-1] # split('.')[0]
    
    return RawName


def process_Source_to_Raw(OneCohort_Args, SourcFile_List, get_RawName_from_SourceFile, SPACE = None):

    RawName_to_dfRaw = {}
    for file_path in SourcFile_List:
        RawName = get_RawName_from_SourceFile(file_path, OneCohort_Args)
        RawName_to_dfRaw[RawName] = file_path


    ### %%%%%%%%%%%%%%%%%%%%% user
    RawName = 'invitation'
    dfRawPath = RawName_to_dfRaw[RawName]
    # print(dfRawPath)
    # df = pd.read_csv(dfRawPath, low_memory=False)
    df = pd.read_parquet(dfRawPath)
    # print(df.shape)
    # df.head()
    ### %%%%%%%%%%%%%%%%%%%%% user

    ### %%%%%%%%%%%%%%%%%%%%% user
    df_inv = df[['patient_id_encoded', 'invitation_id_encoded']]
    # df_inv[['patient_id_encoded', 'invitation_id_encoded']].value_counts()
    ### %%%%%%%%%%%%%%%%%%%%% user


    ### %%%%%%%%%%%%%%%%%%%%% user
    RawName = 'engagement'
    dfRawPath = RawName_to_dfRaw[RawName]
    # df = pd.read_csv(dfRawPath, low_memory=False)
    df = pd.read_parquet(dfRawPath)
    # df.head()
    ### %%%%%%%%%%%%%%%%%%%%% user


    ### %%%%%%%%%%%%%%%%%%%%% user
    # EventName_to_EventNum = df['event_name'].value_counts().sort_index().to_dict()
    # pprint(EventName_to_EventNum)
    ### %%%%%%%%%%%%%%%%%%%%% user

    ### %%%%%%%%%%%%%%%%%%%%% user
    EventName_to_RecNum = {
        'PreviewEvent': 'EgmLink',
        'Prescription Order Page Loaded': 'EgmAuthen', 
        'Call Pharmacy': 'EgmCallPharm',
        'CampaignCopayAssistanceTap': 'EgmSave', # TO confirm. 
        'CampaignEducationalContentTap': 'EgmLearn', 
        'CampaignSchedulePickup': 'EgmRmd', 
        'SendInitialSMSLink': 'SendSMS', 
        'SendReminderSMSLink': 'SendSMS', 
        'SendRenewalSMSLink': 'SendSMS', 
    }

    df['RecName'] = df['event_name'].map(EventName_to_RecNum)
    RecName_to_RecNum = df['RecName'].value_counts().to_dict()
    # RecName_to_RecNum
    ### %%%%%%%%%%%%%%%%%%%%% user

    ### %%%%%%%%%%%%%%%%%%%%% user
    # for RecName, dfx in df.groupby('RecName'):
    #     print(RecName)
    #     df_r = dfx[['campaign_id', 'content_id']].fillna('NoContentID').value_counts().reset_index()
    #     display(df_r)
    ### %%%%%%%%%%%%%%%%%%%%% user


    ### %%%%%%%%%%%%%%%%%%%%% user
    RecName_to_CmpList = {}
    RawName_to_dfRaw = RawName_to_dfRaw

    for RecName in RecName_to_RecNum:
        # print()
        # print(RecName, RecName_to_RecNum[RecName])
        path = os.path.join(OneCohort_Args['FolderPath'], f'{RecName}__rawdata.parquet')
        # print(path)
        RawName_to_dfRaw[RecName] = path
        df_Rec = df[df['RecName'] == RecName].reset_index(drop=True)

        campaign_id_list = list(df_Rec['campaign_id'].unique())
        content_id_list = list(df_Rec['content_id'].unique())
        campaign_content_id_list = list((df_Rec['campaign_id'].astype(str) + '-' + df_Rec['content_id'].astype(str)).unique())
        d = {
            'campaign_id_list': campaign_id_list,
            'content_id_list': content_id_list,
            'campaign_content_id_list': campaign_content_id_list,
        }
        RecName_to_CmpList[RecName] = d
        for k in d: print(k, len(d[k]), d[k])
        # print('campaign_id_list:', len(campaign_id_list))
        # df_Rec.to_csv(path, index=False)
        df_Rec.to_parquet(path)
        
    ### %%%%%%%%%%%%%%%%%%%%% user



    ### %%%%%%%%%%%%%%%%%%%%% user
    RecName = 'Rx'
    path = os.path.join(OneCohort_Args['FolderPath'], f'{RecName}__rawdata.parquet')
    RawName_to_dfRaw[RecName] = path

    RawName = 'prescription'
    dfRawPath = RawName_to_dfRaw[RawName]
    # df = pd.read_csv(dfRawPath, low_memory=False)  
    df = pd.read_parquet(dfRawPath)
    print(df.shape, '<--- prescription record number')
    df = pd.merge(df, df_inv, on = ['patient_id_encoded', 'invitation_id_encoded'])
    print(df.shape, '<--- only keep the prescription from known invitation.')
    df = df.groupby('prescription_id_encoded').first().reset_index()
    print(df.shape, '<--- one prescription id, only with one prescription record.')
    df = df.drop_duplicates()
    print(df.shape)
    ### %%%%%%%%%%%%%%%%%%%%% user

    ### %%%%%%%%%%%%%%%%%%%%% user
    for RecName, CmpListDict in RecName_to_CmpList.items():
        CmpList = CmpListDict['campaign_content_id_list']
        CmpList = [i for i in CmpList if pd.isna(i) == False and i != 'nan-nan']
        if len(CmpList) == 0: continue
        print(RecName, len(CmpList), CmpList)
        df['campaign_content_id'] = df['campaign_id'].astype(str) + '-' + df['content_id'].astype(str)
        df[RecName.replace('Egm', 'Cmp')] = df['campaign_content_id'].isin(CmpList).astype(int)
    # print(df[['CmpRmd', 'CmpLearn', 'CmpSave']].sum())
    # print(path)
    
    if 'state_code' not in df.columns: df['state_code'] = None
        
    ### %%%%%%%%%%%%%%%%%%%%% user
    # Split Rx and P
    patient_id_columns = ['patient_id_encoded']
    patient_related_columns = ['patient_gender', 'patient_age_bucket', 'patient_zipcode_3', 'state_code']

    #-----
    
    #-----
    df_Rx = df[[i for i in df.columns if i not in patient_related_columns]]
    df_PatRec = df[patient_id_columns + patient_related_columns]
    ### %%%%%%%%%%%%%%%%%%%%% user

    ### %%%%%%%%%%%%%%%%%%%%% user
    df_Rec = df
    # print(df_Rec.shape)
    # df_Rec.to_csv(path, index=False)
    df_Rec.to_parquet(path)
    ### %%%%%%%%%%%%%%%%%%%%% user


    ### %%%%%%%%%%%%%%%%%%%%% user
    df = df_PatRec.drop_duplicates()
    df = df.groupby('patient_id_encoded').first().reset_index() # .value_counts()
    df = df.rename(columns = {
        'patient_zipcode_3': 'zipcode3', 
        'patient_zipcode_5': 'zipcode5', 
        })

    if 'zipcode3' in df.columns:
        df['zipcode3'] = df['zipcode3'].astype(str).apply(lambda x: x.split('.')[0])
    if 'zipcode5' in df.columns:
        df['zipcode5'] = df['zipcode5'].astype(str).apply(lambda x: x.split('.')[0])


    if 'timezone' not in df.columns and 'zipcode3' in df.columns:
        ##### everytime, it will read it. 
        # print('here')
        zipcode3_geo_path = os.path.join(SPACE['DATA_EXTERNAL'], 'zipcode3/zipcode3_to_geo.pkl')
        df_zip3 = pd.read_pickle(zipcode3_geo_path)
        df = pd.merge(df, df_zip3[['zipcode3', 'timezone', 'UserTimeZoneOffset']], how = 'left', on = 'zipcode3')

    elif 'timezone' not in df.columns and 'zipcode5' in df.columns:
        zipcode3_geo_path = os.path.join(SPACE['DATA_EXTERNAL'], 'zipcode5/zipcode5_to_geo.pkl')
        df_zip3 = pd.read_pickle(zipcode3_geo_path)
        df = pd.merge(df, df_zip3[['zipcode5', 'timezone', 'UserTimeZoneOffset']], how = 'left', on = 'zipcode5')
    else:
        df['timezone'] = None 
        df['UserTimeZoneOffset'] = None 
    
    df_Rec = df

    RecName = 'Ptt'
    path = os.path.join(OneCohort_Args['FolderPath'], f'{RecName}__rawdata.parquet')
    RawName_to_dfRaw[RecName] = path
    # print(df_Rec.shape)
    df_Rec.to_parquet(path)
    ### %%%%%%%%%%%%%%%%%%%%% user

    return RawName_to_dfRaw


MetaDict = {
	"OneCohort_Args": OneCohort_Args,
	"SourceFile_SuffixList": SourceFile_SuffixList,
	"get_RawName_from_SourceFile": get_RawName_from_SourceFile,
	"process_Source_to_Raw": process_Source_to_Raw
}