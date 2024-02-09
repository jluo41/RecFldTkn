import pandas as pd

import numpy as np

def get_caseobs_id(case_example, CaseObsName):
    PIDValue = case_example['PID']
    ObsDTValue = case_example['ObsDT'].isoformat()
    RxIDValue = case_example['RxID']
    return f'{PIDValue}-{ObsDTValue}-{RxIDValue}'


def get_selected_columns(RecObs_Name, column_names, cohort_args, rec_args, CaseTkn):
    # column_names = [i for i in column_names if 'Tkn_' not in i]
    base_columns = rec_args['RecIDChain'] + [rec_args['RecDT']]
    return base_columns


def get_casetkn_vocab(RecObsName_to_RecObsInfo):
    ############################
    idx2tkn = ['unk', 
               'InvBtnEgm', 'InvBtnEgmClicks', 'InvMinsUntil1stEgm', 
               'RxBtnEgm', 'RxBtnEgmClicks', 'RxMinsUntil1stEgm' ]
    ############################

    tid2tkn = {tid: tkn for tid, tkn in enumerate(idx2tkn)}
    tkn2tid = {tkn: tid for tid, tkn in tid2tkn.items()}
    CaseTknVocab = {}
    CaseTknVocab['tkn'] = {'tid2tkn': tid2tkn, 'tkn2tid': tkn2tid}
    return CaseTknVocab


def fn_CaseTkn(case_example, RecObsName_to_RecObsDS, RecObsName_to_RecObsInfo, CaseTknVocab):
    # input: RecObsName_to_RecObsDS, RecObsName_to_RecObsInfo
    # output: CaseObservation
    assert len(RecObsName_to_RecObsDS) == 1
    RecObsName = [i for i in RecObsName_to_RecObsDS][0]
    RecObsDS   = RecObsName_to_RecObsDS[RecObsName]

    if RecObsDS is None or len(RecObsDS) == 0:
        result_dict = {'InvBtnEgm': 0, 
                       'RxBtnEgm': 0}
    else: 
        RxIDValue = case_example['RxID']
        ObsDTValue = case_example['ObsDT']
        df_InvEgm = RecObsDS.to_pandas()
        time_until_firstclick = (df_InvEgm['DT'].min() - ObsDTValue).total_seconds() / 60
        result_dict = {'InvBtnEgm': 1,  
                       'InvBtnEgmClicks': len(df_InvEgm), 
                       'InvMinsUntil1stEgm': time_until_firstclick}
        # print(df_InvEgm)
        # print(case_example['RxID'])
        df_RxEgm = df_InvEgm[df_InvEgm['RxID'] == RxIDValue].reset_index()
        if len(df_RxEgm) == 0:
            result_dict['RxBtnEgm'] = 0
        else:
            time_until_firstclick = (df_RxEgm['DT'].min() - ObsDTValue).total_seconds() / 60
            result_dict['RxBtnEgm'] = 1
            result_dict['RxBtnEgmClicks'] = len(df_RxEgm)
            result_dict['RxMinsUntil1stEgm'] = time_until_firstclick
            
    result_dict = {'tkn': [i for i in result_dict], 'wgt': [float(round(result_dict[i], 2)) for i in result_dict]}
    result_dict['tid'] = [CaseTknVocab['tkn']['tkn2tid'][tkn] for tkn in result_dict['tkn']]
    CaseObservation = {'tid': result_dict['tid'], 'wgt': result_dict['wgt']}
    return CaseObservation


MetaDict = {
	"get_caseobs_id": get_caseobs_id,
	"get_selected_columns": get_selected_columns,
	"get_casetkn_vocab": get_casetkn_vocab,
	"fn_CaseTkn": fn_CaseTkn
}