from recfldtkn.obsname import parse_RecObsName

import pandas as pd

import numpy as np

def get_caseobs_id(case_example, CaseObsName):
    RecordName = CaseObsName.split('_')[0].split('-')[0].replace('ro.', '')
    RID = RecordName + 'ID'
    assert RID in case_example.keys() 
    RIDVlaue = case_example[RID]
    return f'{RID}:{RIDVlaue}'


def get_selected_columns(RecObs_Name, column_names, cohort_args, rec_args, CaseTkn):
    from recfldtkn.obsname import parse_RecObsName
    # print(RecName)
    if 'RecDT' in rec_args:
        base_columns = rec_args['RecIDChain'] + [rec_args['RecDT']]
    else:
        base_columns = rec_args['RecIDChain']
    # print(base_columns)
    FldName = parse_RecObsName(RecObs_Name)['FldName']
    # print(FldName)
    FldNameTkn = FldName + 'Tkn'
    key_list = [i for i in column_names if FldNameTkn in i]
    # print(key_list)
    selected_columns = base_columns + key_list
    return selected_columns


def get_casetkn_vocab(RecObsName_to_RecObsInfo):

    RecFldName_list = list(set([RecObsInfo['RecName'] + '-' +RecObsInfo['FldName'] 
                                for RecObsName, RecObsInfo in RecObsName_to_RecObsInfo.items()
                                if RecObsInfo['FldName'] is not None ]
                                ))
    assert len(RecFldName_list) == 1

    ############################ tkn 
    RecObsName = [i for i in RecObsName_to_RecObsInfo][0]
    RecObsInfo = RecObsName_to_RecObsInfo[RecObsName]
    idx2tkn = RecObsInfo['FldIdx2Tkn']   
    tid2tkn = {tid: tkn for tid, tkn in enumerate(idx2tkn)}
    tkn2tid = {tkn: tid for tid, tkn in tid2tkn.items()}
    CaseTknVocab = {}
    CaseTknVocab['tkn'] = {'tid2tkn': tid2tkn, 'tkn2tid': tkn2tid}
    ############################

    ############################ 5MinLoc: TODO
    return CaseTknVocab


def fn_CaseTkn(case_example, RecObsName_to_RecObsDS, RecObsName_to_RecObsInfo, CaseTknVocab):
    # input: RecObsName_to_RecObsDS, RecObsName_to_RecObsInfo
    # output: CaseObservation
    d = {}

    ObsDTValue = case_example['ObsDT']

    assert len(RecObsName_to_RecObsDS) == 1
    RecObsName = [i for i in RecObsName_to_RecObsDS][0]
    RecObsDS   = RecObsName_to_RecObsDS[RecObsName]
    RecObsInfo = RecObsName_to_RecObsInfo[RecObsName]
    # RecDT = RecObsInfo['rec_args']['RecDT']
    # CkpdInfo = RecObsInfo['CkpdInfo']  
    # StartIdx5Min = CkpdInfo['StartIdx5Min']
    # EndIdx5Min = CkpdInfo['EndIdx5Min']
        
    # 1. get a subset of a Record Type: e.g., CGM5Min_Bf24H, with TknIdx.
    # further filter
    ds_p_ckpd_rec_fld = RecObsDS
    df = ds_p_ckpd_rec_fld.to_pandas()
    
    RecID = RecObsInfo['rec_args']['RecID']
    assert RecID in case_example 
    df = df[df[RecID] == case_example[RecID]].reset_index(drop=True)
    assert len(df) == 1
    record_dict = df.iloc[0].to_dict()

    FldName = parse_RecObsName(RecObsName)['FldName']
    FldNameTkn = FldName + 'Tkn'
    # print(RecObsName)
    # print(df.columns)
    # print([i for i in df.columns if FldNameTkn in i])
    key_list = [i for i in df.columns if FldNameTkn in i]

    record = {k.split('_')[-1]: list(v) for k, v in record_dict.items() if k in key_list}
    CaseObservation = {'tid': record['tknidx'], 'wgt': record['wgt']} 
    
    return CaseObservation


MetaDict = {
	"get_caseobs_id": get_caseobs_id,
	"get_selected_columns": get_selected_columns,
	"get_casetkn_vocab": get_casetkn_vocab,
	"fn_CaseTkn": fn_CaseTkn
}