import pandas as pd

import numpy as np

def get_caseobs_id(case_example, CaseObsName):
    PIDValue = case_example['PID']
    ObsDTValue = case_example['ObsDT'].isoformat()
    return f'{PIDValue}&{ObsDTValue}'


def get_selected_columns(RecObs_Name, column_names, cohort_args, rec_args, CaseTkn):
    # RecName = RecObs_Name.split('-')[0]
    base_columns = [cohort_args['RootID'], rec_args['RecID'], rec_args['RecDT']]
    return base_columns


def get_casetkn_vocab(RecObsName_to_RecObsInfo):
    # phi_list
    RecFldName_list = list(set([RecObsInfo['RecName'] + '-' +RecObsInfo['FldName'] 
                                for RecObsName, RecObsInfo in RecObsName_to_RecObsInfo.items()
                                if RecObsInfo['FldName'] is not None ]
                                ))
    
    # print('RecFldName_list----->', RecFldName_list)
    # for current version: for any Phi, we only process one or zero phi. 
    assert len(RecFldName_list) <= 1

    ############################
    idx2tkn = ['[UNK]', 'recnum', 'recspan']
    ############################

    tid2tkn = {tid: tkn for tid, tkn in enumerate(idx2tkn)}
    tkn2tid = {tkn: tid for tid, tkn in tid2tkn.items()}
    CaseTknVocab = {}
    CaseTknVocab['tkn'] = {'tid2tkn': tid2tkn, 'tkn2tid': tkn2tid}
    return CaseTknVocab


def fn_CaseTkn(case_example, 
               RecObsName_to_RecObsDS, 
               RecObsName_to_RecObsInfo, 
               CaseTknVocab):
    # input: RecObsName_to_RecObsDS, RecObsName_to_RecObsInfo
    # output: CaseObservation
    d = {}
    assert len(RecObsName_to_RecObsDS) == 1
 
    # RO
    RO = list(RecObsName_to_RecObsDS.keys())[0]
    # for RecObsName in RecObsName_to_RecObsDS:
    
    #############################################
    ROds   = RecObsName_to_RecObsDS[RO] # dataframe: RecObsName: Rx-bf24.. RecObsDS: the df: record collection
    ROInfo = RecObsName_to_RecObsInfo[RO]
    
    RecDT = ROInfo['rec_args']['RecDT']
    if ROds is not None:
        DT_s_obs = ROds[ 0][RecDT] # the time of first records
        DT_e_obs = ROds[-1][RecDT] # pd.to_datetime(dates[idx_e-1]) # the last one smaller than idx_e
        d['recnum'] = len(ROds)
        d['recspan'] = (DT_e_obs - DT_s_obs).total_seconds() / 60 # + 5
    else:
        d['recnum'] = 0
        d['recspan'] = 0
    #############################################

    tkn2tid = CaseTknVocab['tkn']['tkn2tid']
    tkn = [i for i in d.keys()]
    tid = [tkn2tid[i] for i in tkn]
    wgt = [d[i] for i in tkn]
    CO = {'tid': tid, 'wgt': wgt}
    # CaseObservation
    return CO


MetaDict = {
	"get_caseobs_id": get_caseobs_id,
	"get_selected_columns": get_selected_columns,
	"get_casetkn_vocab": get_casetkn_vocab,
	"fn_CaseTkn": fn_CaseTkn
}