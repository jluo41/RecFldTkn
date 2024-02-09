
# from recfldtkn.constaidatatools import  pipeline_for_ai_dataset_construction as pipeline_case
import os
import tokenizers
from tokenizers.pre_tokenizers import WhitespaceSplit
import pandas as pd
from datetime import datetime 
from typing import List
from datasets import concatenate_datasets
from datasets import DatasetDict
import os
import datasets
import numpy as np 
from .configfn import load_record_args, load_fldtkn_args
from .loadtools import load_module_variables
from .obsname import parse_RecObsName, convert_RecObsName_and_CaseTkn_to_CaseObsName
from .configfn import load_cohort_args
from .obsname import convert_RecObsName_and_CaseTkn_to_CaseObsName
from .loadtools import load_module_variables, update_args_to_list
from .observer import get_RecObsName_to_RecObsInfo, CaseObserverTransformer
from .obsname import parse_RecObsName


def fetch_caseobs_Phi_tools(CaseTkn, CaseObsName, SPACE):
    # step 4: prepare the \Phi's tools
    # print(f'\n================ Prepare the \Phi Tools: CaseTkn-{args.case_tkn} ================')
    pypath = os.path.join(SPACE['CODE_FN'], 'fn_casetkn', f'casetkn_{CaseTkn}.py')
    # print('PyPath:', pypath)
    module = load_module_variables(pypath)
    get_caseobs_id = module.get_caseobs_id
    get_selected_columns = module.get_selected_columns
    get_casetkn_vocab = module.get_casetkn_vocab
    fn_CaseTkn = module.fn_CaseTkn
    CaseObsFolder = os.path.join(SPACE['DATA_CaseObs'], CaseObsName)
    # if not os.path.exists(CaseObsFolder): os.makedirs(CaseObsFolder)
    # print(CaseObsFolder)
    return get_caseobs_id, get_selected_columns, get_casetkn_vocab, fn_CaseTkn, CaseObsFolder


def fetch_casetaskop_Gamma_tools(CaseTaskOp, SPACE):
    # step 4: prepare the \Gamma's tools

    pypath = os.path.join(SPACE['CODE_FN'], 'fn_casetaskop',  f'casetaskop_{CaseTaskOp}.py')
    print('PyPath:', pypath)
    module = load_module_variables(pypath)
    BATCHED = int(module.BATCHED)
    get_case_op_vocab = module.get_case_op_vocab
    fn_Case_Operation_Tkn = module.fn_Case_Operation_Tkn

    return BATCHED, get_case_op_vocab, fn_Case_Operation_Tkn


def pipeline_caseset_to_caseobservation(ds_case, 
                                        Record_Observations_List, 
                                        CaseTkn, 
                                        SPACE, 
                                        cohort_args, 
                                        record_to_ds_rec = {}, 
                                        record_to_ds_rec_info = {}, 
                                        use_caseobs_from_disk = True, 
                                        batch_size = 1000):
    
    CaseObsName = convert_RecObsName_and_CaseTkn_to_CaseObsName(Record_Observations_List, CaseTkn)
    get_caseobs_id, get_selected_columns, get_casetkn_vocab, fn_CaseTkn, CaseObsFolder = fetch_caseobs_Phi_tools(CaseTkn, CaseObsName, SPACE)

    # print('\n======== generate RecCkpd_to_RecObsInfo: the RecObsInfo for each Rec_Ckpd =========')
    RecObsName_to_RecObsInfo = get_RecObsName_to_RecObsInfo(Record_Observations_List, 
                                                            CaseTkn, 
                                                            get_selected_columns,
                                                            cohort_args, 
                                                            cohort_args['Ckpd_ObservationS'], 
                                                            record_to_ds_rec, 
                                                            record_to_ds_rec_info)

    # initialize the CaseObserverTransformer to a Phi pipeline.
    df_case = ds_case.to_pandas()
    caseobs_ids = set(df_case.apply(lambda x: get_caseobs_id(x, CaseObsName), axis = 1).unique())

    fn_caseobs_Phi = CaseObserverTransformer(RecObsName_to_RecObsInfo, 
                                             CaseTkn, 
                                             get_casetkn_vocab, 
                                             fn_CaseTkn, 
                                             get_caseobs_id,
                                             use_caseobs_from_disk, 
                                             CaseObsFolder, # should you include the CaseFolder information here?
                                             caseobs_ids)
    
    # get the CaseTknVocab out. 
    CaseTknVocab = fn_caseobs_Phi.CaseTknVocab 

    # start = datetime.now()
    ds_caseobs = ds_case.map(fn_caseobs_Phi, 
                             batched = True, 
                             batch_size= batch_size, 
                             load_from_cache_file=False, 
                             new_fingerprint = CaseObsName,
                             # desc = CaseObsName, # do not use this, it will cause the error. I don't know why.
                            )
    # end = datetime.now()
    return RecObsName_to_RecObsInfo, ds_caseobs, fn_caseobs_Phi, CaseTknVocab


def convert_case_observations_to_co_to_observation(case_observations):
    co_to_CaseObsName = {i.split(':')[0]: i.split(':')[1] for i in case_observations}
    co_to_CaseObsNameInfo = {}
    for co, CaseObsName in co_to_CaseObsName.items():
        # print(co, CaseObsName)
        Record_Observations_List = CaseObsName.split('_')[0].replace('ro.', '').split('&')
        CaseTkn = CaseObsName.split('_')[1].replace('ct.', '')
        # print(CaseTkn)
        co_to_CaseObsNameInfo[co] = {'Record_Observations_List': Record_Observations_List, 'CaseTkn': CaseTkn}
    return co_to_CaseObsName, co_to_CaseObsNameInfo


def get_RecNameList_and_FldTknList(co_to_CaseObsNameInfo):
    RecNameList_All = []
    CkpdNameList_All = []
    FldTknList_All = []
    CaseTknList = []

    for co, CaseObsNameInfo in co_to_CaseObsNameInfo.items():
        Record_Observations_List = CaseObsNameInfo['Record_Observations_List']
        CaseTkn = CaseObsNameInfo['CaseTkn']
        RecNameList = [parse_RecObsName(i)['RecName'] for i in Record_Observations_List]
        CkpdNameList = [parse_RecObsName(i)['CkpdName'] for i in Record_Observations_List]
        FldNameList = [parse_RecObsName(i)['RecName'] + '-' + parse_RecObsName(i)['FldName'] 
                       for i in Record_Observations_List if parse_RecObsName(i)['FldName'] is not None]
        
        RecNameList_All = RecNameList_All + [i for i in RecNameList if i is not None]
        CkpdNameList_All = CkpdNameList_All + [i for i in CkpdNameList if i is not None]
        FldTknList_All = FldTknList_All + [i for i in FldNameList if i is not None]
        CaseTknList.append(CaseTkn)
        # print(co, RecNameList, CkpdNameList, FldNameList, CaseTkn)

    d = {'RecNameList': list(set(RecNameList_All)), 
         'CkpdNameList': list(set(CkpdNameList_All)), 
         'FldTknList': list(set(FldTknList_All)), 
         'CaseTknList': list(set(CaseTknList))}
    return d


def pipeline_to_generate_co_to_CaseObsInfo(ds_case, 
                                           co_to_CaseObsNameInfo, 
                                           SPACE, 
                                           cohort_args,
                                           case_id_columns, 
                                           record_to_ds_rec, 
                                           record_to_ds_rec_info,
                                           use_caseobs_from_disk,
                                           batch_size = 1000):

    co_to_CaseObsInfo = {}
    for co, CaseObsNameInfo in co_to_CaseObsNameInfo.items():
        CaseObsInfo = {}
        Record_Observations_List = CaseObsNameInfo['Record_Observations_List']
        CaseTkn = CaseObsNameInfo['CaseTkn']
        # print('\n\n=============\n')
        print(Record_Observations_List, CaseTkn)
        RecObsName_to_RecObsInfo, ds_caseobs, fn_caseobs_Phi, CaseTknVocab = pipeline_caseset_to_caseobservation(ds_case, 
                                                                                                                 Record_Observations_List, 
                                                                                                                 CaseTkn, 
                                                                                                                 SPACE, 
                                                                                                                 cohort_args, 
                                                                                                                 record_to_ds_rec, 
                                                                                                                 record_to_ds_rec_info, 
                                                                                                                 use_caseobs_from_disk,
                                                                                                                 batch_size)
        # print(RecObsName_to_RecObsInfo)
        if len(fn_caseobs_Phi.CaseTknVocab) > 0 and use_caseobs_from_disk == True: 
            CaseObsName = fn_caseobs_Phi.CaseObsName
            fn_caseobs_Phi.save_new_caseobs_to_ds_caseobs()
            pd.DataFrame({CaseObsName: CaseTknVocab}).to_pickle(fn_caseobs_Phi.CaseObsFolder_vocab)
        
        # print(ds_caseobs)
        # print(fn_caseobs_Phi)
        # print(CaseTknVocab)
        columns = [i for i in ds_caseobs.column_names if i not in case_id_columns]
        for column in columns:
            ds_caseobs = ds_caseobs.rename_column(column, co + '_' + column)
        CaseObsInfo['ds_caseobs'] = ds_caseobs
        CaseObsInfo['vocab_caseobs'] = CaseTknVocab
        CaseObsInfo['RecObsName_to_RecObsInfo'] = RecObsName_to_RecObsInfo
        co_to_CaseObsInfo[co] = CaseObsInfo
    return co_to_CaseObsInfo


def get_complete_dataset(co_to_CaseObsInfo):
    # 1. merge the datasets together 
    ds = None 
    for idx, co in enumerate(co_to_CaseObsInfo):
        CaseObsInfo = co_to_CaseObsInfo[co]
        ds_caseobs = CaseObsInfo['ds_caseobs']
        if idx == 0:
            ds = ds_caseobs 
        else:
            assert len(ds) == len(ds_caseobs)
            columns = [i for i in ds_caseobs.column_names if i not in ds.column_names]
            for column in columns:
                ds = ds.add_column(column, ds_caseobs[column]) 
    # print(ds) 
    return ds


def pipeline_case(ds_case_dict,     # C
                  case_observations,# CO 
                  CaseTaskOp,       # Gamma
                  case_id_columns,
                  cohort_args, 
                  record_to_ds_rec, 
                  record_to_ds_rec_info,
                  use_caseobs_from_disk,
                  SPACE):

    co_to_CaseObsName, co_to_CaseObsNameInfo = convert_case_observations_to_co_to_observation(case_observations)
    # print(co_to_CaseObsName)
    # pprint(co_to_CaseObsNameInfo, sort_dicts=False)
    pipeline_info_dict = get_RecNameList_and_FldTknList(co_to_CaseObsNameInfo)
    # pprint(pipeline_info_dict)

    d = {}
    for case_type, ds_case in ds_case_dict.items():
        print(f'\n\n================ case_type: {case_type} ================')
        co_to_CaseObsInfo = pipeline_to_generate_co_to_CaseObsInfo(ds_case, 
                                                                    co_to_CaseObsNameInfo, 
                                                                    SPACE, 
                                                                    cohort_args,
                                                                    case_id_columns, 
                                                                    record_to_ds_rec, 
                                                                    record_to_ds_rec_info,
                                                                    use_caseobs_from_disk,
                                                                    batch_size = 1000)
        ds = get_complete_dataset(co_to_CaseObsInfo)
        # print(ds)
        d[case_type] = ds

    co_to_CaseObsVocab = {co: CaseObsInfo['vocab_caseobs'] for co, CaseObsInfo in co_to_CaseObsInfo.items()}
    co_to_RecObsName_to_RecObsInfo = {co: CaseObsInfo['RecObsName_to_RecObsInfo'] for co, CaseObsInfo in co_to_CaseObsInfo.items()}
    ds_caseobslist_dict = datasets.DatasetDict(d)

    BATCHED, get_case_op_vocab, fn_Case_Operation_Tkn = fetch_casetaskop_Gamma_tools(CaseTaskOp, SPACE)
    CaseTaskOpVocab = get_case_op_vocab(co_to_CaseObsVocab)
    ds_case_proc = ds_caseobslist_dict.map(lambda examples: fn_Case_Operation_Tkn(examples, co_to_CaseObsVocab, CaseTaskOpVocab), 
                                               batch_size=1000,
                                               batched = BATCHED)
    
    old_columns = list(ds_caseobslist_dict.column_names.values())[0]
    ds_case_proc = ds_case_proc.remove_columns(old_columns)


    results = {}
    results['co_to_CaseObsName'] = co_to_CaseObsName
    results['co_to_CaseObsNameInfo'] = co_to_CaseObsNameInfo
    results['pipeline_info_dict'] = pipeline_info_dict
    results['co_to_CaseObsVocab'] = co_to_CaseObsVocab
    results['co_to_RecObsName_to_RecObsInfo'] = co_to_RecObsName_to_RecObsInfo
    results['ds_caseobslist_dict'] = ds_caseobslist_dict
    results['CaseTaskOpVocab'] = CaseTaskOpVocab
    results['ds_case_proc'] = ds_case_proc
    return results


def create_tokenizer(CaseTaskOpVocab, tokenizer_folder):

    tokenizer_dict = {}
    for sequence_name in CaseTaskOpVocab:

        tid2tkn = CaseTaskOpVocab[sequence_name]['tid2tkn']
        tkn2tid = {v: k for k, v in tid2tkn.items()}
        unk_token = tid2tkn[0]
        if 'unk' not in unk_token:
            print(f'Warning: unk_token: {unk_token} is not unk')

        tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab = tkn2tid, unk_token=unk_token))
        tokenizer.pre_tokenizer = WhitespaceSplit()

        if not os.path.exists(tokenizer_folder):
            os.makedirs(os.path.dirname(tokenizer_folder))

        tokenizer_path = os.path.join(tokenizer_folder, sequence_name + '.json')
        if os.path.exists(tokenizer_path):
            tokenizer_old = tokenizers.Tokenizer.from_file(tokenizer_path)
            old_dict = tokenizer_old.get_vocab()    
            new_dict = tokenizer.get_vocab()
            diff1 = old_dict.keys() - new_dict.keys()
            diff2 = new_dict.keys() - old_dict.keys()
            assert len(diff1) == 0 and len(diff2) == 0
        else:
            tokenizer.save(tokenizer_path)

        tokenizer_dict[sequence_name] = tokenizer
    return tokenizer_dict
            

# pipeline_case = pipeline_for_ai_dataset_construction