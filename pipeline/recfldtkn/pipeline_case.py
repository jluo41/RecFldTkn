import os
import logging
import datasets
import tokenizers
import numpy as np 

import pandas as pd
from typing import List
from datetime import datetime 
from datasets import concatenate_datasets, DatasetDict
from tokenizers.pre_tokenizers import WhitespaceSplit

from .obsname import convert_case_observations_to_co_to_observation, get_RecNameList_and_FldTknList
from .observer import get_RecObsName_to_RecObsInfo, CaseObserverTransformer
from .pipeline_model import get_Trigger_Cases, convert_TriggerCases_to_LearningCases
from .pipeline_model import assign_caseSplitTag_to_dsCaseLearning
from .loadtools import fetch_trigger_tools 
from .observer import get_fn_case_GammaFullInfo
from .loadtools import fetch_entry_tools, load_module_variables

logger = logging.getLogger(__name__)


def get_dfset_from_SetName(df_dsmp, SetName, case_id_columns, SubGroupFilterMethod):
    Split, SubGroup = SetName.split(':')
    for i in ['In', 'Out', 'Train', 'Valid', 'Test']:
        if i.lower() in Split.lower():
            df_dsmp = df_dsmp[df_dsmp[i]].reset_index(drop = True)
    df_set = df_dsmp[case_id_columns].reset_index(drop = True)
    if SubGroup.lower() != 'all':
        pass
    return df_set


def pipeline_from_dsRec_to_dsCase(data_args, 
                                  base_config, SPACE, 
                                  RecName_to_dsRec, RecName_to_dsRecInfo, 
                                  use_CO_from_disk, use_CF_from_disk,
                                  use_inference = False):

    # part 1: case args
    case_args = data_args['case_args']
    if use_inference == False:
        cohort_label_list = case_args['cohort_label_list']
    else:
        cohort_label_list = [9] # 9 is the inference cohort label.


    # step 1.1: get Trigger Cases
    TriggerCaseMethod = case_args['TriggerCaseMethod']
    df_case = get_Trigger_Cases(TriggerCaseMethod,
                                cohort_label_list, base_config, SPACE, 
                                RecName_to_dsRec, RecName_to_dsRecInfo)
    
    # step 1.2: convert Trigger Cases to Learning Cases
    Trigger2LearningMethods = case_args['Trigger2LearningMethods']
    if use_inference == True:
        Trigger2LearningMethods = [v for v in Trigger2LearningMethods if v['type'] != 'learning-only']

    df_case_learning = convert_TriggerCases_to_LearningCases(df_case, 
                                                            cohort_label_list,
                                                            Trigger2LearningMethods, 
                                                            base_config, 
                                                            use_inference,)

    # step 1.3: convert df_case_learning to ds_case_dict with splitmethod
    if use_inference == True:
        ds_case_dict = {'inference': datasets.Dataset.from_pandas(df_case_learning)}
    else:
        split_args = data_args['split_args']
        df_case = df_case_learning.copy()
        df_case = assign_caseSplitTag_to_dsCaseLearning(df_case, 
                                                        split_args['RANDOM_SEED'], 
                                                        split_args['downsample_ratio'], 
                                                        split_args['out_ratio'], 
                                                        split_args['test_ratio'], 
                                                        split_args['valid_ratio'])

        set_args = data_args['set_args']
        TrainSetName = set_args['TrainSetName']
        EvalSetNames = set_args['EvalSetNames']
        SubGroupFilterMethod = set_args['SubGroupFilterMethod']
        case_id_columns = Trigger_Tools['case_id_columns']
        Trigger_Tools = fetch_trigger_tools(TriggerCaseMethod, SPACE)
        ds_case_dict = {}
        for SetName in [TrainSetName] + EvalSetNames:
            df_set = get_dfset_from_SetName(df_case, SetName, case_id_columns, SubGroupFilterMethod)
            ds_set = datasets.Dataset.from_pandas(df_set)
            ds_case_dict[SetName] = ds_set
        ds_case_dict = datasets.DatasetDict(ds_case_dict) 
    
    # part 2: data point
    datapoint_args = data_args['datapoint_args']
    CFType_to_CaseFeatInfo = {}
    for CFType, Gamma_Config in datapoint_args.items():
        
        if use_inference == True and 'output' in CFType.lower(): continue 
        
        logger.info(f'============ CFType: {CFType} =============')
        CaseFeatInfo = get_fn_case_GammaFullInfo(Gamma_Config, 
                                                base_config, 
                                                RecName_to_dsRec, 
                                                RecName_to_dsRecInfo, 
                                                df_case_learning,
                                                use_CF_from_disk,
                                                use_CO_from_disk)
        
        CFType_to_CaseFeatInfo[CFType] = CaseFeatInfo
        FnCaseFeatGamma = CaseFeatInfo['FnCaseFeatGamma']
        batch_size = CaseFeatInfo.get('batch_size', 1000)
        CaseFeatName = CaseFeatInfo['CaseFeatName']
        for splitname, ds_caseset in ds_case_dict.items():
            logger.info(f'----- splitname: {splitname} -----')
            ds_caseset = ds_caseset.map(FnCaseFeatGamma, 
                                        batched = True, 
                                        batch_size= batch_size, 
                                        load_from_cache_file = False, 
                                        new_fingerprint = CaseFeatName + splitname.replace(':', '_'))
            ds_case_dict[splitname] = ds_caseset

        if len(FnCaseFeatGamma.new_CFs) > 0 and use_CF_from_disk == True:
            logger.info(f'----- Save CF {CaseFeatName}: to Cache File -----')
            FnCaseFeatGamma.save_new_CFs_to_disk()
        
        for COName, FnCaseObsPhi in FnCaseFeatGamma.COName_to_FnCaseObsPhi.items():
            if len(FnCaseObsPhi.new_COs) > 0 and use_CO_from_disk == True:
                logger.info(f'----- Save CO {COName}: to Cache File -----')
                FnCaseObsPhi.save_new_COs_to_disk()

        
    # part 3: entry process
    entry_args = data_args['entry_args']
    Entry_Tools = fetch_entry_tools(entry_args, SPACE)
    fn_entry_method_for_input = Entry_Tools['entry_method_for_input']
    fn_entry_method_for_output = Entry_Tools['entry_method_for_output']
    fn_entry_method_for_finaldata = Entry_Tools['entry_method_for_finaldata']

    ds_casefinal_dict = {}
    for split, dataset in ds_case_dict.items():
        dataset = fn_entry_method_for_finaldata(dataset, 
                                                CFType_to_CaseFeatInfo,
                                                fn_entry_method_for_input, 
                                                fn_entry_method_for_output,
                                                use_inference)
        ds_casefinal_dict[split] = dataset
    return ds_casefinal_dict



# ------------------------------------ pipeline for one case ------------------------------------ #
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
    
    Phi_Tools = fetch_caseobs_Phi_tools(CaseTkn, CaseObsName, SPACE)
    # print('\n======== generate RecCkpd_to_RecObsInfo: the RecObsInfo for each Rec_Ckpd =========')
    
    get_selected_columns = Phi_Tools['get_selected_columns']
    RecObsName_to_RecObsInfo = get_RecObsName_to_RecObsInfo(Record_Observations_List, 
                                                            CaseTkn, 
                                                            get_selected_columns,
                                                            cohort_args, 
                                                            cohort_args['Ckpd_ObservationS'], 
                                                            record_to_ds_rec, 
                                                            record_to_ds_rec_info)

    # initialize the CaseObserverTransformer to a Phi pipeline.
    df_case = ds_case.to_pandas()
    get_caseobs_id = Phi_Tools['get_caseobs_id']
    caseobs_ids = set(df_case.apply(lambda x: get_caseobs_id(x, CaseObsName), axis = 1).unique())

    # create the mapping functions to get the CO. 
    get_casetkn_vocab = Phi_Tools['get_casetkn_vocab']
    fn_CaseTkn = Phi_Tools['fn_CaseTkn']
    CaseObsFolder = Phi_Tools['CaseObsFolder']
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

    # do the case observation.
    ds_caseobs = ds_case.map(fn_caseobs_Phi, 
                             batched = True, 
                             batch_size= batch_size, 
                             load_from_cache_file=False, 
                             new_fingerprint = CaseObsName)
    

    # save the new caseobs to the disk if necessary
    if len(fn_caseobs_Phi.new_calculated_caseobs) > 0 and use_caseobs_from_disk == True: 
        # save the new caseobs to the disk.
        fn_caseobs_Phi.save_new_caseobs_to_ds_caseobs()
        # save the vocab to the disk.
        CaseObsFolder_vocab = fn_caseobs_Phi.CaseObsFolder_vocab
        df_Vocab = pd.DataFrame({CaseObsName: CaseTknVocab})
        df_Vocab.to_pickle(CaseObsFolder_vocab)
        
    data_CaseObs_Results = {}
    data_CaseObs_Results['ds_caseobs'] = ds_caseobs
    data_CaseObs_Results['CaseObsName'] = CaseObsName
    data_CaseObs_Results['RecObsName_to_RecObsInfo'] = RecObsName_to_RecObsInfo
    data_CaseObs_Results['fn_caseobs_Phi'] = fn_caseobs_Phi
    data_CaseObs_Results['CaseTknVocab'] = CaseTknVocab
    return data_CaseObs_Results


# ------------------------------------ pipeline for a case list ------------------------------------ #
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
        
        logger.info(f'CaseObsNameInfo: {Record_Observations_List}, {CaseTkn}')

        old_columns = list(ds_case.column_names)
        # print('check whether ds_case will change along the way ----------->', ds_case)
        data_CaseObs_Results = pipeline_caseset_to_caseobservation(ds_case, 
                                                                   Record_Observations_List, 
                                                                   CaseTkn, 
                                                                   SPACE, 
                                                                   cohort_args, 
                                                                   record_to_ds_rec, 
                                                                   record_to_ds_rec_info, 
                                                                   use_caseobs_from_disk,
                                                                   batch_size)

        # select columns and update column names
        ds_caseobs = data_CaseObs_Results['ds_caseobs']
        CaseTknVocab = data_CaseObs_Results['CaseTknVocab']
        RecObsName_to_RecObsInfo = data_CaseObs_Results['RecObsName_to_RecObsInfo']

        new_columns = [i for i in ds_caseobs.column_names if i not in old_columns]
        ds_caseobs = ds_caseobs.select_columns(case_id_columns + new_columns)
        for column in new_columns:
            ds_caseobs = ds_caseobs.rename_column(column, co + '_' + column)

        # save information to CaseObsInfo
        CaseObsInfo['ds_caseobs'] = ds_caseobs
        CaseObsInfo['CaseObsName'] = data_CaseObs_Results['CaseObsName']
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
    PipelineInfo = get_RecNameList_and_FldTknList(co_to_CaseObsNameInfo)

    d = {}
    for case_type, ds_case in ds_case_dict.items():
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
        d[case_type] = ds

    co_to_CaseObsVocab = {co: CaseObsInfo['vocab_caseobs'] for co, CaseObsInfo in co_to_CaseObsInfo.items()}
    co_to_RecObsName_to_RecObsInfo = {co: CaseObsInfo['RecObsName_to_RecObsInfo'] for co, CaseObsInfo in co_to_CaseObsInfo.items()}
    ds_caseobslist_dict = datasets.DatasetDict(d)

    Gamma_tools = fetch_casetaskop_Gamma_tools(CaseTaskOp, SPACE)
    get_case_op_vocab = Gamma_tools['get_case_op_vocab']
    CaseTaskOpVocab = get_case_op_vocab(co_to_CaseObsVocab)
    SeqTypeList = Gamma_tools.get('SeqTypeList', ['input_ids', 'labels'])
    fn_Case_Operation_Tkn = Gamma_tools['fn_Case_Operation_Tkn']
    ds_case_proc = ds_caseobslist_dict.map(lambda examples: fn_Case_Operation_Tkn(examples, co_to_CaseObsVocab, CaseTaskOpVocab), 
                                           batch_size = Gamma_tools.get('batch_size', 1000), 
                                           batched = Gamma_tools.get('BATCHED', 1))
    
    
    results = {}
    results['PipelineInfo'] = PipelineInfo
    results['co_to_RecObsName_to_RecObsInfo'] = co_to_RecObsName_to_RecObsInfo
    results['co_to_CaseObsName'] = co_to_CaseObsName
    results['co_to_CaseObsNameInfo'] = co_to_CaseObsNameInfo
    results['co_to_CaseObsVocab'] = co_to_CaseObsVocab
    results['ds_caseobslist_dict'] = ds_caseobslist_dict
    results['CaseTaskOpVocab'] = CaseTaskOpVocab
    results['SeqTypeList'] = SeqTypeList
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
            # os.makedirs(os.path.dirname(tokenizer_folder))
            os.makedirs(tokenizer_folder)

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
