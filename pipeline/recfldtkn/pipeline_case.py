import os
import logging


import numpy as np 
import pandas as pd

from typing import List
from datetime import datetime 

from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ProcessPoolExecutor, as_completed

import datasets
from datasets import concatenate_datasets, DatasetDict
from datasets.fingerprint import Hasher

import tokenizers
from tokenizers.pre_tokenizers import WhitespaceSplit

from .obsname import convert_case_observations_to_co_to_observation, get_RecNameList_and_FldTknList
from .observer import get_RecObsName_to_RecObsInfo, CaseObserverTransformer
from .loadtools import fetch_trigger_tools 
from .observer import get_fn_case_GammaFullInfo
from .loadtools import load_module_variables
from .loadtools import load_ds_rec_and_info


# logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s:%(asctime)s:(%(filename)s@%(lineno)d %(name)s)]: %(message)s')



def get_tkn_value(x, tkn_id, ids, wgts):
    if tkn_id in x[ids]:
        index = list(x[ids]).index(tkn_id)
        return x[wgts][index]
    else:
        return 0
    

def load_complete_PipelineInfo(datapoint_args, cohort_config, use_inference):

    case_observations_total = []
    for k, v in datapoint_args.items():
        if use_inference == True and 'Output' in k: continue 
        case_observations_total = case_observations_total + v['case_observations']

    _, co_to_CaseObsNameInfo = convert_case_observations_to_co_to_observation(case_observations_total)
    PipelineInfo = get_RecNameList_and_FldTknList(co_to_CaseObsNameInfo)
    PipelineInfo['FldTknList'] = [i+'Tkn' for i in PipelineInfo['FldTknList']]

    # 3. get record_sequence
    record_sequence = PipelineInfo['RecNameList']
    RecName_to_PrtRecName = cohort_config['RecName_to_PrtRecName']
    record_sequence_prt = [RecName_to_PrtRecName[recname] for recname in record_sequence if RecName_to_PrtRecName[recname] != 'None']
    # print(record_sequence_prt)
    new_records = [i for i in record_sequence_prt if i not in record_sequence]
    while len(new_records) > 0:
        record_sequence.extend(new_records)
        record_sequence_prt = [RecName_to_PrtRecName[recname] for recname in record_sequence if RecName_to_PrtRecName[recname] != 'None']
        new_records = [i for i in record_sequence_prt if i not in record_sequence]
    record_sequence = [recname for recname in cohort_config['RecName_Sequence'] if recname in PipelineInfo['RecNameList']]

    PipelineInfo['RecNameList'] = record_sequence # 
    return PipelineInfo


def get_ds_case_to_process(InputCaseSetName, 
                           cohort_label_list, 
                           TriggerCaseMethod, 
                           cohort_config, 
                           SPACE,
                           RecName_to_dsRec = {},
                           RecName_to_dsRecInfo = {}):
    '''
        this function is designed to be used in the training pipeline only.
        We only want to get a ds_case to tag, filter, split (optional), training, and inference. 
    '''
    if InputCaseSetName is None: 
        COHORT = 'C' + '.'.join([str(i) for i in cohort_label_list])
        TRIGGER = TriggerCaseMethod
        InputCaseSetName = '-'.join([COHORT, TRIGGER])
        
    InputCaseFolder = os.path.join(SPACE['DATA_CaseSet'], InputCaseSetName)
    InputCaseFile = InputCaseFolder + '.p'

        
    if os.path.exists(InputCaseFolder):
        assert os.path.isdir(InputCaseFolder)
        L = []
        if 'TaggingSize' in InputCaseFolder:
            tag_method_list = sorted(os.listdir(InputCaseFolder))
            for tag_method in tag_method_list:
                file_list = sorted(os.listdir(os.path.join(InputCaseFolder, tag_method)))
                df_case_tag = pd.concat([pd.read_pickle(os.path.join(InputCaseFolder, tag_method, f)) 
                                        for f in file_list])
                df_case_tag = df_case_tag.reset_index(drop = True)
                L.append(df_case_tag)
            pypath = os.path.join(cohort_config['trigger_pyfolder'], f'{TriggerCaseMethod}.py')
            module = load_module_variables(pypath)
            case_id_columns = module.case_id_columns
            df_case = L[0]
            for df_case_tag in L[1:]:
                assert len(df_case) == len(df_case_tag)
                columns = [i for i in df_case_tag.columns if i not in df_case.columns]
                df_case = pd.merge(df_case, df_case_tag[case_id_columns + columns], on=case_id_columns)
        else:
            file_list = sorted(os.listdir(InputCaseFolder))
            df_case = pd.concat([pd.read_pickle(os.path.join(InputCaseFolder, f)) for f in file_list])

    elif os.path.exists(InputCaseFile):
        assert os.path.exists(InputCaseFile)
        df_case = pd.read_pickle(InputCaseFile)
    else: 
        ################## get df_case to tag ##################
        # cohort_config = load_cohort_args(recfldtkn_config_path, SPACE)
        Trigger_Tools = fetch_trigger_tools(TriggerCaseMethod, SPACE)
        case_id_columns = Trigger_Tools['case_id_columns']
        special_columns = Trigger_Tools['special_columns'] 
        TriggerRecName = Trigger_Tools['TriggerRecName']
        convert_TriggerEvent_to_Caseset = Trigger_Tools['convert_TriggerEvent_to_Caseset']
        
        if TriggerRecName in RecName_to_dsRec:
            ds_rec = RecName_to_dsRec[TriggerRecName]
        else:
            ds_rec, _ = load_ds_rec_and_info(TriggerRecName, cohort_config, cohort_label_list)
        
        # ds_rec, _ = load_ds_rec_and_info(TriggerRecName, cohort_config, cohort_label_list)
        df_case = convert_TriggerEvent_to_Caseset(ds_rec, case_id_columns, special_columns, cohort_config)
        InputCaseFile = InputCaseFolder + '.p'
        df_case.to_pickle(InputCaseFile)
    # print(df_case.shape)
    return InputCaseSetName, df_case
    

def fn_casefeat_querying(ds_case, 
                         cohort_config, 
                         case_id_columns,
                         case_observations, 
                         name_CaseGamma, 
                         RecName_to_dsRec = {},
                         RecName_to_dsRecInfo = {}, 
                         use_CF_from_disk = True, 
                         use_CO_from_disk = True):
    
    #############################
    Gamma_Config = {
        'case_observations':case_observations,
        'name_CaseGamma': name_CaseGamma, # CF
    }
    #############################

    # case_id_columns = cohort_config['case_id_columns']
    df_case = ds_case.select_columns(case_id_columns).to_pandas()
    CaseFeatInfo = get_fn_case_GammaFullInfo(Gamma_Config, 
                                            cohort_config, 
                                            RecName_to_dsRec, 
                                            RecName_to_dsRecInfo, 
                                            df_case,
                                            use_CF_from_disk,
                                            use_CO_from_disk)
    
    FnCaseFeatGamma = CaseFeatInfo['FnCaseFeatGamma']
    batch_size = CaseFeatInfo.get('batch_size', 1000)
    CaseFeatName = CaseFeatInfo['CaseFeatName']
    # CF_vocab = CaseFeatInfo['CF_vocab']
    ds_case = ds_case.map(FnCaseFeatGamma, 
                            batched = True, 
                            batch_size= batch_size, 
                            load_from_cache_file = False, 
                            new_fingerprint = CaseFeatName)
    
    if len(FnCaseFeatGamma.new_CFs) > 0 and use_CF_from_disk == True:
        logger.info(f'----- Save CF {CaseFeatName}: to Cache File -----')
        FnCaseFeatGamma.save_new_CFs_to_disk()
    
    for COName, FnCaseObsPhi in FnCaseFeatGamma.COName_to_FnCaseObsPhi.items():
        if len(FnCaseObsPhi.new_COs) > 0 and use_CO_from_disk == True:
            logger.info(f'----- Save CO {COName}: to Cache File -----')
            FnCaseObsPhi.save_new_COs_to_disk()

    return CaseFeatInfo, ds_case


######### tagging tasks #########
def process_df_tagging_tasks(df_case, 
                             cohort_label_list,
                             case_id_columns,
                             InputCaseSetName, 
                             TagMethod_List, 
                             cf_to_QueryCaseFeatConfig,  
                             cohort_config,
                             SPACE, 
                             RecName_to_dsRec, 
                             RecName_to_dsRecInfo,
                             use_CF_from_disk, 
                             use_CO_from_disk, 
                             chunk_id = None, 
                             start_idx = None, 
                             end_idx = None, 
                             chunk_size = 500000,
                             save_to_pickle = False,
                             ):
    
    # You can also check whether it is in disk or not.
    OutputCaseSetName = '-'.join([InputCaseSetName, 't.'+'.'.join(TagMethod_List)])
    
    for TagMethod in TagMethod_List:
        # logger.info(f'--------- TagMethod {TagMethod} -------------')
        # logger.info(f'--------- before tagging {df_case.shape} -------------')
        if TagMethod in cf_to_QueryCaseFeatConfig:
            QueryCaseFeatConfig = cf_to_QueryCaseFeatConfig[TagMethod]
            TagMethodFile = TagMethod + '.' + Hasher.hash(QueryCaseFeatConfig)
        else:
            TagMethodFile = TagMethod

        ############################### option 1: loading from disk ###############################
        if save_to_pickle == True:
            Folder = os.path.join(SPACE['DATA_CaseSet'], InputCaseSetName + '-Tagging' + f'Size{chunk_size}')
            if chunk_id == None: chunk_id = 0
            if start_idx == None: start_idx = 0
            if end_idx == None: end_idx = len(df_case)

            start_idx_k = start_idx // 1000
            end_idx_k = end_idx // 1000
            filename = f'idx{chunk_id:05}_{start_idx_k:06}k_{end_idx_k:06}k.p'
            fullfilepath = os.path.join(Folder, TagMethodFile, filename)
            fullfolderpath = os.path.dirname(fullfilepath)

            if not os.path.exists(fullfolderpath): 
                os.makedirs(fullfolderpath)


            # print(fullfilepath)
            # print(os.path.exists(fullfilepath))
            if os.path.exists(fullfilepath): 
                df_case_new = pd.read_pickle(fullfilepath) 
                columns = [i for i in df_case_new.columns if i not in df_case.columns]
                df_case = pd.merge(df_case, df_case_new[case_id_columns + columns], on=case_id_columns)
                continue 

        ############################### option 2: calculating and saving to disk ###############################
        if TagMethod in cf_to_QueryCaseFeatConfig:
            QueryCaseFeatConfig = cf_to_QueryCaseFeatConfig[TagMethod]
            case_observations = QueryCaseFeatConfig['case_observations']
            name_CaseGamma = QueryCaseFeatConfig['name_CaseGamma']
            tkn_name_list = QueryCaseFeatConfig['tkn_name_list']
            
            ds_case = datasets.Dataset.from_pandas(df_case)
            CaseFeatInfo, ds_case = fn_casefeat_querying(ds_case, 
                                                         cohort_config, 
                                                         case_id_columns,
                                                         case_observations, 
                                                         name_CaseGamma, 
                                                         RecName_to_dsRec,
                                                         RecName_to_dsRecInfo, 
                                                         use_CF_from_disk, 
                                                         use_CO_from_disk)
            
            # assert name_CaseGamma is a special CaseGamma
            CF_vocab = CaseFeatInfo['CF_vocab']
            CaseFeatName = CaseFeatInfo['CaseFeatName']
            rename_dict = {i: CaseFeatName + ':' + i for i in CF_vocab}
            df_case = ds_case.to_pandas().rename(columns=rename_dict)

            
            ids  = CaseFeatName + ':' + 'input_ids'
            wgts = CaseFeatName + ':' + 'input_wgts'
            for tkn_name in tkn_name_list:
                tkn_id = CF_vocab['input_ids']['tkn2tid'][tkn_name]
                df_case[tkn_name] = df_case.apply(lambda x: get_tkn_value(x, tkn_id, ids, wgts), axis = 1)
            df_case = df_case.drop(columns = [ids, wgts])
            
        else:
            pypath = os.path.join(SPACE['CODE_FN'], 'fn_learning', f'{TagMethod}.py')
            module = load_module_variables(pypath)
            MetaDict = module.MetaDict
            # print([i for i in MetaDict])
            if 'InfoRecName' in MetaDict:
                # print('error')
                InfoRecName, subgroup_columns, fn_case_tagging = module.InfoRecName, module.subgroup_columns, module.fn_case_tagging
                ds_info, _ = load_ds_rec_and_info(InfoRecName, cohort_config, cohort_label_list)
                logger.info(f'ds_info is {InfoRecName}: {ds_info}')
                df_case = fn_case_tagging(df_case, ds_info, subgroup_columns, cohort_config)
            elif 'fn_case_tagging_on_casefeat' in MetaDict:
                fn_case_tagging_on_casefeat = MetaDict['fn_case_tagging_on_casefeat']
                df_case = fn_case_tagging_on_casefeat(df_case)
            else:
                raise ValueError('No fn_case_tagging_on_casefeat or InfoRecName in the module')
            
        if save_to_pickle == True:
            df_case.to_pickle(fullfilepath)

    return OutputCaseSetName, df_case
    

def _process_chunk_tagging(chunk_id, df_case, chunk_size, 
                           cohort_label_list, case_id_columns,
                           InputCaseSetName, TagMethod_List, cf_to_QueryCaseFeatConfig, cohort_config,
                           SPACE, RecName_to_dsRec, RecName_to_dsRecInfo, 
                           use_CF_from_disk, use_CO_from_disk, save_to_pickle):
    start_idx = chunk_id * chunk_size
    end_idx = min((chunk_id + 1) * chunk_size, len(df_case))
    df_case_chunk = df_case.iloc[start_idx:end_idx].reset_index(drop=True)
    # print(f'chunk_id: {chunk_id}, start_idx: {start_idx}, end_idx: {end_idx}')
    _, df_case_chunk_tagged = process_df_tagging_tasks(
        df_case_chunk, 
        cohort_label_list,
        case_id_columns,
        InputCaseSetName, 
        TagMethod_List, 
        cf_to_QueryCaseFeatConfig,  
        cohort_config,
        SPACE, 
        RecName_to_dsRec, 
        RecName_to_dsRecInfo,
        use_CF_from_disk, 
        use_CO_from_disk,
        chunk_id, 
        start_idx, 
        end_idx, 
        chunk_size,
        save_to_pickle,
    )
    return df_case_chunk_tagged


def process_df_tagging_tasks_in_chunks(df_case, 
                                       cohort_label_list,
                                       case_id_columns,
                                       InputCaseSetName, 
                                       TagMethod_List,
                                       cf_to_QueryCaseFeatConfig,
                                       cohort_config, 
                                       SPACE, 
                                       RecName_to_dsRec, 
                                       RecName_to_dsRecInfo,
                                       use_CF_from_disk,
                                       use_CO_from_disk,
                                       start_chunk_id = 0, 
                                       end_chunk_id = None, 
                                       chunk_size = 500000,
                                       save_to_pickle = True, 
                                       num_processors = 0):
    
    # --------------- get the tagging method name ---------------
    OutputCaseSetName = '-'.join([InputCaseSetName, 'Tagging'])
    Folder = os.path.join(SPACE['DATA_CaseSet'], OutputCaseSetName + f'Size{chunk_size}')
    
    # --------------- process tagging tasks ---------------
    if end_chunk_id is None: end_chunk_id = len(df_case) // chunk_size + 1
    chunk_id_list = range(start_chunk_id, end_chunk_id)



    df_case_chunk_tagged_list = []
    if num_processors > 1:
        # with ProcessPoolExecutor(max_workers=num_processors) as executor:
        #     df_case_chunk_tagged_list = list(executor.map(process_chunk, chunk_id_list))
        with ProcessPoolExecutor(max_workers=num_processors) as executor:
            futures = [executor.submit(
                        _process_chunk_tagging, 
                            chunk_id, df_case, chunk_size, 
                            cohort_label_list, case_id_columns,
                            InputCaseSetName, TagMethod_List, cf_to_QueryCaseFeatConfig, cohort_config,
                            SPACE, RecName_to_dsRec, RecName_to_dsRecInfo, 
                            use_CF_from_disk, use_CO_from_disk, save_to_pickle) 
                
                for chunk_id in chunk_id_list]

            for future in as_completed(futures):
                df_case_chunk_tagged_list.append(future.result())

    else:
        for chunk_id in chunk_id_list:
            # print(chunk_id)
            df_case_chunk_tagged = _process_chunk_tagging(
                                        chunk_id, df_case, chunk_size, 
                                        cohort_label_list, case_id_columns,
                                        InputCaseSetName, TagMethod_List, cf_to_QueryCaseFeatConfig, cohort_config,
                                        SPACE, RecName_to_dsRec, RecName_to_dsRecInfo, 
                                        use_CF_from_disk, use_CO_from_disk, save_to_pickle)
            df_case_chunk_tagged_list.append(df_case_chunk_tagged)

    df_case_tagged_final = pd.concat(df_case_chunk_tagged_list, axis = 0).reset_index(drop = True)
    OutputCaseSetName_final = '-'.join([InputCaseSetName, 't.'+'.'.join(TagMethod_List)])
    return OutputCaseSetName_final, df_case_tagged_final


######### casefeat tasks #########
def process_df_casefeat_tasks(df_case, 
                              cohort_label_list,
                              case_id_columns,
                              InputCaseSetName, 
                              CaseFeat_List, 
                              cf_to_CaseFeatConfig, 
                              cohort_config,
                              SPACE, 
                              RecName_to_dsRec, 
                              RecName_to_dsRecInfo,
                              use_CF_from_disk, 
                              use_CO_from_disk, 
                              chunk_id, 
                              start_idx, 
                              end_idx, 
                              chunk_size,
                              save_to_pickle):
    
    ds_case = datasets.Dataset.from_pandas(df_case)
    cf_to_CaseFeatInfo = {}
    for CaseFeat in CaseFeat_List:
        logger.info(f'--------- CaseFeat {CaseFeat} -------------')

        assert  CaseFeat in cf_to_CaseFeatConfig
        CaseFeatConfig = cf_to_CaseFeatConfig[CaseFeat]
        case_observations = CaseFeatConfig['case_observations']
        name_CaseGamma = CaseFeatConfig['name_CaseGamma']
        old_columns = ds_case.column_names
        CaseFeatInfo, ds_case = fn_casefeat_querying(ds_case, 
                                                     cohort_config, 
                                                     case_id_columns,
                                                     case_observations, 
                                                     name_CaseGamma, 
                                                     RecName_to_dsRec,
                                                     RecName_to_dsRecInfo, 
                                                     use_CF_from_disk, 
                                                     use_CO_from_disk)
        
        new_columns = [i for i in ds_case.column_names if i not in old_columns]
        rename_dict = {i: CaseFeat + '.' + i for i in new_columns}
        for old_name, new_name in rename_dict.items():
            ds_case = ds_case.rename_column(old_name, new_name)
        cf_to_CaseFeatInfo[CaseFeat] = CaseFeatInfo
    # OutputCaseSetName = '-'.join([InputCaseSetName, 'cf.'+'.'.join(CaseFeat_List)])
    return cf_to_CaseFeatInfo, ds_case


def _process_chunk_casefeat(chunk_id, df_case, chunk_size, 
                           cohort_label_list, case_id_columns,
                           InputCaseSetName, CaseFeat_List, cf_to_CaseFeatConfig, cohort_config,
                           SPACE, RecName_to_dsRec, RecName_to_dsRecInfo, 
                           use_CF_from_disk, use_CO_from_disk, save_to_pickle):
    start_idx = chunk_id * chunk_size
    end_idx = min((chunk_id + 1) * chunk_size, len(df_case))
    df_case_chunk = df_case.iloc[start_idx:end_idx].reset_index(drop=True)
    _, df_case_chunk_casefeat = process_df_casefeat_tasks(
        df_case_chunk, 
        cohort_label_list,
        case_id_columns,
        InputCaseSetName, 
        CaseFeat_List, 
        cf_to_CaseFeatConfig,  
        cohort_config,
        SPACE, 
        RecName_to_dsRec, 
        RecName_to_dsRecInfo,
        use_CF_from_disk, 
        use_CO_from_disk,
        chunk_id, 
        start_idx, 
        end_idx, 
        chunk_size,
        save_to_pickle,
    )
    return df_case_chunk_casefeat


def process_df_casefeat_tasks_in_chunks(df_case, 
                                        cohort_label_list,
                                        case_id_columns,
                                        InputCaseSetName, 
                                        CaseFeat_List, 
                                        cf_to_CaseFeatConfig, 
                                        cohort_config,
                                        SPACE, 
                                        RecName_to_dsRec, 
                                        RecName_to_dsRecInfo,
                                        use_CF_from_disk, 
                                        use_CO_from_disk,
                                        start_chunk_id = 0, 
                                        end_chunk_id = None, 
                                        chunk_size = 500000,
                                        save_to_pickle = False, 
                                        num_processors = 1):
    
    # --------------- cf_to_CaseFeatInfo ---------------
    cf_to_CaseFeatInfo = {}
    for CaseFeat in CaseFeat_List:
        assert  CaseFeat in cf_to_CaseFeatConfig
        CaseFeatConfig = cf_to_CaseFeatConfig[CaseFeat]
        case_observations = CaseFeatConfig['case_observations']
        name_CaseGamma = CaseFeatConfig['name_CaseGamma']
        #############################
        Gamma_Config = {
            'case_observations':case_observations,
            'name_CaseGamma': name_CaseGamma, # CF
        }
        #############################
        # df_case = ds_case.select_columns(case_id_columns).to_pandas()
        CaseFeatInfo = get_fn_case_GammaFullInfo(Gamma_Config, 
                                                 cohort_config, 
                                                 RecName_to_dsRec, 
                                                 RecName_to_dsRecInfo, 
                                                 df_case[case_id_columns],
                                                 use_CF_from_disk,
                                                 use_CO_from_disk)
        cf_to_CaseFeatInfo[CaseFeat] = CaseFeatInfo
        
    
    # --------------- get the tagging method name ---------------
    OutputCaseSetName = '-'.join([InputCaseSetName, 'Tagging'])
    Folder = os.path.join(SPACE['DATA_CaseSet'], OutputCaseSetName + f'Size{chunk_size}')

    # --------------- process tagging tasks ---------------
    if end_chunk_id is None: end_chunk_id = len(df_case) // chunk_size + 1
    chunk_id_list = range(start_chunk_id, end_chunk_id)

    ds_case_chunk_casefeat_list = []
    if num_processors > 1:
        # with ProcessPoolExecutor(max_workers=num_processors) as executor:
        #     df_case_chunk_tagged_list = list(executor.map(process_chunk, chunk_id_list))
        with ProcessPoolExecutor(max_workers = num_processors) as executor:
            futures = [executor.submit(
                        _process_chunk_casefeat, 
                            chunk_id, df_case, chunk_size, 
                           cohort_label_list, case_id_columns,
                           InputCaseSetName, CaseFeat_List, cf_to_CaseFeatConfig, cohort_config,
                           SPACE, RecName_to_dsRec, RecName_to_dsRecInfo, 
                           use_CF_from_disk, use_CO_from_disk, save_to_pickle) 
                
                for chunk_id in chunk_id_list]

            for future in as_completed(futures):
                ds_case_chunk_casefeat_list.append(future.result())
                # future.result()
    else:
        for chunk_id in chunk_id_list:
            ds_case_chunk_casefeat = _process_chunk_casefeat(
                                        chunk_id, df_case, chunk_size, 
                                        cohort_label_list, case_id_columns,
                                        InputCaseSetName, CaseFeat_List, cf_to_CaseFeatConfig, cohort_config,
                                        SPACE, RecName_to_dsRec, RecName_to_dsRecInfo, 
                                        use_CF_from_disk, use_CO_from_disk, save_to_pickle
                                    ) 
            ds_case_chunk_casefeat_list.append(ds_case_chunk_casefeat)
                            
    ds_case = concatenate_datasets(ds_case_chunk_casefeat_list)
    return cf_to_CaseFeatInfo, ds_case


######### split tasks #########
def generate_random_tags(df_case, RANDOM_SEED, RootID, ObsDT):
    np.random.seed(RANDOM_SEED)
    # RootID, ObsDT = 'PID', 'ObsDT'

    # down sample 
    df_case['RandDownSample'] = np.random.rand(len(df_case))

    # in&out
    df_P = df_case[[RootID]].drop_duplicates().reset_index(drop = True)
    df_P['RandInOut'] = np.random.rand(len(df_P))
    df_case = pd.merge(df_case, df_P)

    # test
    df_case['CaseLocInP'] = df_case.groupby(RootID).cumcount()
    df_case = pd.merge(df_case, df_case[RootID].value_counts().reset_index())
    df_case['CaseRltLocInP'] = df_case['CaseLocInP'] /  df_case['count']
    
    # test other options
    df_case['RandTest'] = np.random.rand(len(df_case))

    # validation
    df_case['RandValidation'] = np.random.rand(len(df_case))

    df_case = df_case.drop(columns = ['CaseLocInP', 'count']).reset_index(drop = True)
    df_case = df_case.sort_values('RandDownSample').reset_index(drop = True)

    random_columns = ['RandDownSample', 'RandInOut', 'CaseRltLocInP', 'RandTest', 'RandValidation']
    return df_case, random_columns


def assign_caseSplitTag_to_dsCase(df_case, 
                                RANDOM_SEED, 
                                RootID, 
                                ObsDT,
                                downsample_ratio, 
                                out_ratio, 
                                test_ratio, 
                                valid_ratio):

    df = df_case 
    df_rs, random_columns = generate_random_tags(df, RANDOM_SEED, RootID, ObsDT,)
    df_dsmp = df_rs[df_rs['RandDownSample'] <= downsample_ratio].reset_index(drop = True)

    df_dsmp['Out'] = df_dsmp['RandInOut'] < out_ratio
    df_dsmp['In'] = df_dsmp['RandInOut'] >= out_ratio
    assert df_dsmp[['Out', 'In']].sum(axis = 1).mean() == 1

    if 'tail' in str(test_ratio):
        TestSelector = 'CaseRltLocInP'
        test_ratio = float(test_ratio.replace('tail', ''))
        test_threshold = 1 - test_ratio
    elif type(test_ratio) != float and type(test_ratio) != int:
        TestSelector = 'ObsDT'
        test_threshold = pd.to_datetime(test_ratio)
    else:
        TestSelector = 'RandTest'
        test_threshold = 1 - test_ratio

    if 'tail' in str(valid_ratio):
        ValidSelector = 'CaseRltLocInP'
        valid_ratio = float(valid_ratio.replace('tail', ''))
        valid_threshold = 1 - valid_ratio
    elif type(valid_ratio) != float and type(valid_ratio) != int:
        ValidSelector = 'ObsDT'
        valid_threshold = pd.to_datetime(valid_ratio)
    else:
        ValidSelector = 'RandTest' 
        valid_threshold = 1 - valid_ratio
        
    df_dsmp['Test'] = df_dsmp[TestSelector] > test_threshold
    df_dsmp['Valid'] = (df_dsmp[ValidSelector] > valid_threshold) & (df_dsmp['Test'] == False)
    df_dsmp['Train'] = (df_dsmp['Test'] == False) & (df_dsmp['Valid'] == False)

    assert df_dsmp[['Train', 'Valid', 'Test']].sum(axis = 1).mean() == 1

    df_dsmp = df_dsmp.drop(columns = random_columns)
    return df_dsmp



######### filtering tasks #########
def process_df_filtering_tasks(df_case, FilterMethod_List, SPACE):
    for FilterMethod in FilterMethod_List:
        logger.info(f'FilterMethod: {FilterMethod}')
        pypath = os.path.join(SPACE['CODE_FN'], 'fn_learning', f'{FilterMethod}.py')
        module = load_module_variables(pypath)

        fn_case_filtering = module.fn_case_filtering
        logger.info(f'before filtering: {df_case.shape}')
        df_case = fn_case_filtering(df_case)
        logger.info(f'after filtering: {df_case.shape}')
    return df_case

######### inference tasks #########

def get_dfset_from_SetName(df_dsmp, SetName, case_id_columns, SubGroupFilterMethod):
    df_dsmp = df_dsmp.copy()
    Split, SubGroup = SetName.split('_')
    for i in ['In', 'Out', 'Train', 'Valid', 'Test']:
        if i.lower() in [t.lower() for t in Split.split('-')]:
            df_dsmp = df_dsmp[df_dsmp[i]].reset_index(drop = True)
    df_set = df_dsmp[case_id_columns].reset_index(drop = True)
    if SubGroup.lower() != 'all': pass
    return df_set

# def get_dfset_from_SetName(df_dsmp, SetName, case_id_columns, SubGroupFilterMethod):
#     Split, SubGroup = SetName.split(':')
#     for i in ['In', 'Out', 'Train', 'Valid', 'Test']:
#         if i.lower() in Split.lower():
#             df_dsmp = df_dsmp[df_dsmp[i]].reset_index(drop = True)
#     df_set = df_dsmp[case_id_columns].reset_index(drop = True)
#     if SubGroup.lower() != 'all':
#         pass
#     return df_set


# def generate_casefeat_to_datasetdict(Gamma_Configs, 
#                                      ds_case_dict,
#                                      df_case_all, 
#                                      cohort_config, 
#                                      RecName_to_dsRec, 
#                                      RecName_to_dsRecInfo, 
#                                      use_CF_from_disk, 
#                                      use_CO_from_disk,
#                                      use_inference):
#     CFType_to_CaseFeatInfo = {}
#     for CFType, Gamma_Config in Gamma_Configs.items():
#         # this need to pay attention
#         if use_inference == True and 'output' in CFType.lower(): continue 
        
#         logger.info(f'============ CFType: {CFType} =============')
#         # df_case = datasets.concatenate_datasets([ds_case_dict[split] for split in ds_case_dict])
#         CaseFeatInfo = get_fn_case_GammaFullInfo(Gamma_Config, 
#                                                 cohort_config, 
#                                                 RecName_to_dsRec, 
#                                                 RecName_to_dsRecInfo, 
#                                                 df_case_all,
#                                                 use_CF_from_disk,
#                                                 use_CO_from_disk)
#         CFType_to_CaseFeatInfo[CFType] = CaseFeatInfo
#         FnCaseFeatGamma = CaseFeatInfo['FnCaseFeatGamma']
#         batch_size = CaseFeatInfo.get('batch_size', 10)
#         CaseFeatName = CaseFeatInfo['CaseFeatName']
#         for splitname, ds_caseset in ds_case_dict.items():
#             logger.info(f'----- splitname: {splitname} -----')
#             ds_caseset = ds_caseset.map(FnCaseFeatGamma, 
#                                         batched = True, 
#                                         batch_size= batch_size, 
#                                         load_from_cache_file = False, 
#                                         new_fingerprint = CaseFeatName + splitname.replace(':', '_'))
#             ds_case_dict[splitname] = ds_caseset
#             # logger.info(ds_caseset)
#         ######## save to cache file #########
#         if len(FnCaseFeatGamma.new_CFs) > 0 and use_CF_from_disk == True:
#             logger.info(f'----- Save CF {CaseFeatName}: to Cache File -----')
#             FnCaseFeatGamma.save_new_CFs_to_disk()
        
#         for COName, FnCaseObsPhi in FnCaseFeatGamma.COName_to_FnCaseObsPhi.items():
#             if len(FnCaseObsPhi.new_COs) > 0 and use_CO_from_disk == True:
#                 logger.info(f'----- Save CO {COName}: to Cache File -----')
#                 FnCaseObsPhi.save_new_COs_to_disk()
#     return ds_case_dict, CaseFeatInfo

# def create_tokenizer_from_CF_vocab(CF_vocab):
#     tokenizer_dict = {}
#     for SeqType in CF_vocab:

#         tid2tkn = {int(tid): tkn for tid, tkn in CF_vocab[SeqType]['tid2tkn'].items()}
#         tkn2tid = {v: k for k, v in tid2tkn.items()}
#         unk_token = tid2tkn[0]
#         if 'unk' not in unk_token:
#             print(f'Warning: unk_token: {unk_token} is not unk')

#         tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab = tkn2tid, unk_token=unk_token))
#         tokenizer.pre_tokenizer = WhitespaceSplit()
#         tokenizer_dict[SeqType] = tokenizer
#     return tokenizer_dict


# def pipeline_from_dsRec_to_dsCase(data_args, 
#                                   cohort_config, SPACE, 
#                                   RecName_to_dsRec, RecName_to_dsRecInfo, 
#                                   use_CO_from_disk, use_CF_from_disk,
#                                   use_inference = False):

#     # part 1: case args
#     case_args = data_args['case_args']
#     if use_inference == False:
#         cohort_label_list = case_args['cohort_label_list']
#     else:
#         cohort_label_list = [9] # 9 is the inference cohort label.


#     # step 1.1: get Trigger Cases
#     TriggerCaseMethod = case_args['TriggerCaseMethod']
#     df_case = get_Trigger_Cases(TriggerCaseMethod,
#                                 cohort_label_list, cohort_config, SPACE, 
#                                 RecName_to_dsRec, RecName_to_dsRecInfo)
    
#     # step 1.2: convert Trigger Cases to Learning Cases
#     Trigger2LearningMethods = case_args['Trigger2LearningMethods']
#     if use_inference == True:
#         Trigger2LearningMethods = [v for v in Trigger2LearningMethods if v['type'] != 'learning-only']

#     df_case_learning = convert_TriggerCases_to_LearningCases(df_case, 
#                                                             cohort_label_list,
#                                                             Trigger2LearningMethods, 
#                                                             cohort_config, 
#                                                             use_inference,)

#     # step 1.3: convert df_case_learning to ds_case_dict with splitmethod
#     if use_inference == True:
#         ds_case_dict = {'inference': datasets.Dataset.from_pandas(df_case_learning)}
#     else:
#         split_args = data_args['split_args']
#         df_case = df_case_learning.copy()
#         df_case = assign_caseSplitTag_to_dsCaseLearning(df_case, 
#                                                         split_args['RANDOM_SEED'], 
#                                                         split_args['downsample_ratio'], 
#                                                         split_args['out_ratio'], 
#                                                         split_args['test_ratio'], 
#                                                         split_args['valid_ratio'])

#         set_args = data_args['set_args']
#         TrainSetName = set_args['TrainSetName']
#         EvalSetNames = set_args['EvalSetNames']
#         SubGroupFilterMethod = set_args['SubGroupFilterMethod']
#         case_id_columns = Trigger_Tools['case_id_columns']
#         Trigger_Tools = fetch_trigger_tools(TriggerCaseMethod, SPACE)
#         ds_case_dict = {}
#         for SetName in [TrainSetName] + EvalSetNames:
#             df_set = get_dfset_from_SetName(df_case, SetName, case_id_columns, SubGroupFilterMethod)
#             ds_set = datasets.Dataset.from_pandas(df_set)
#             ds_case_dict[SetName] = ds_set
#         ds_case_dict = datasets.DatasetDict(ds_case_dict) 
    
#     # part 2: data point
#     datapoint_args = data_args['datapoint_args']
#     CFType_to_CaseFeatInfo = {}
#     for CFType, Gamma_Config in datapoint_args.items():
        
#         if use_inference == True and 'output' in CFType.lower(): continue 
        
#         logger.info(f'============ CFType: {CFType} =============')
#         CaseFeatInfo = get_fn_case_GammaFullInfo(Gamma_Config, 
#                                                 cohort_config, 
#                                                 RecName_to_dsRec, 
#                                                 RecName_to_dsRecInfo, 
#                                                 df_case_learning,
#                                                 use_CF_from_disk,
#                                                 use_CO_from_disk)
        
#         CFType_to_CaseFeatInfo[CFType] = CaseFeatInfo
#         FnCaseFeatGamma = CaseFeatInfo['FnCaseFeatGamma']
#         batch_size = CaseFeatInfo.get('batch_size', 1000)
#         CaseFeatName = CaseFeatInfo['CaseFeatName']
#         for splitname, ds_caseset in ds_case_dict.items():
#             logger.info(f'----- splitname: {splitname} -----')
#             ds_caseset = ds_caseset.map(FnCaseFeatGamma, 
#                                         batched = True, 
#                                         batch_size= batch_size, 
#                                         load_from_cache_file = False, 
#                                         new_fingerprint = CaseFeatName + splitname.replace(':', '_'))
#             ds_case_dict[splitname] = ds_caseset

#         if len(FnCaseFeatGamma.new_CFs) > 0 and use_CF_from_disk == True:
#             logger.info(f'----- Save CF {CaseFeatName}: to Cache File -----')
#             FnCaseFeatGamma.save_new_CFs_to_disk()
        
#         for COName, FnCaseObsPhi in FnCaseFeatGamma.COName_to_FnCaseObsPhi.items():
#             if len(FnCaseObsPhi.new_COs) > 0 and use_CO_from_disk == True:
#                 logger.info(f'----- Save CO {COName}: to Cache File -----')
#                 FnCaseObsPhi.save_new_COs_to_disk()

        
#     # part 3: entry process
#     entry_args = data_args['entry_args']
#     Entry_Tools = fetch_entry_tools(entry_args, SPACE)
#     fn_entry_method_for_input = Entry_Tools['entry_method_for_input']
#     fn_entry_method_for_output = Entry_Tools['entry_method_for_output']
#     fn_entry_method_for_finaldata = Entry_Tools['entry_method_for_finaldata']

#     ds_casefinal_dict = {}
#     for split, dataset in ds_case_dict.items():
#         dataset = fn_entry_method_for_finaldata(dataset, 
#                                                 CFType_to_CaseFeatInfo,
#                                                 fn_entry_method_for_input, 
#                                                 fn_entry_method_for_output,
#                                                 use_inference)
#         ds_casefinal_dict[split] = dataset
#     return ds_casefinal_dict



# # ------------------------------------ pipeline for one case ------------------------------------ #
# def pipeline_caseset_to_caseobservation(ds_case, 
#                                         Record_Observations_List, 
#                                         CaseTkn, 
#                                         SPACE, 
#                                         cohort_args, 
#                                         record_to_ds_rec = {}, 
#                                         record_to_ds_rec_info = {}, 
#                                         use_caseobs_from_disk = True, 
#                                         batch_size = 1000):
    
#     CaseObsName = convert_RecObsName_and_CaseTkn_to_CaseObsName(Record_Observations_List, CaseTkn)
    
#     Phi_Tools = fetch_caseobs_Phi_tools(CaseTkn, CaseObsName, SPACE)
#     # print('\n======== generate RecCkpd_to_RecObsInfo: the RecObsInfo for each Rec_Ckpd =========')
    
#     get_selected_columns = Phi_Tools['get_selected_columns']
#     RecObsName_to_RecObsInfo = get_RecObsName_to_RecObsInfo(Record_Observations_List, 
#                                                             CaseTkn, 
#                                                             get_selected_columns,
#                                                             cohort_args, 
#                                                             cohort_args['Ckpd_ObservationS'], 
#                                                             record_to_ds_rec, 
#                                                             record_to_ds_rec_info)

#     # initialize the CaseObserverTransformer to a Phi pipeline.
#     df_case = ds_case.to_pandas()
#     get_caseobs_id = Phi_Tools['get_caseobs_id']
#     caseobs_ids = set(df_case.apply(lambda x: get_caseobs_id(x, CaseObsName), axis = 1).unique())

#     # create the mapping functions to get the CO. 
#     get_casetkn_vocab = Phi_Tools['get_casetkn_vocab']
#     fn_CaseTkn = Phi_Tools['fn_CaseTkn']
#     CaseObsFolder = Phi_Tools['CaseObsFolder']
#     fn_caseobs_Phi = CaseObserverTransformer(RecObsName_to_RecObsInfo, 
#                                              CaseTkn, 
#                                              get_casetkn_vocab, 
#                                              fn_CaseTkn, 
#                                              get_caseobs_id,
#                                              use_caseobs_from_disk, 
#                                              CaseObsFolder, # should you include the CaseFolder information here?
#                                              caseobs_ids)
    
#     # get the CaseTknVocab out. 
#     CaseTknVocab = fn_caseobs_Phi.CaseTknVocab 

#     # do the case observation.
#     ds_caseobs = ds_case.map(fn_caseobs_Phi, 
#                              batched = True, 
#                              batch_size= batch_size, 
#                              load_from_cache_file=False, 
#                              new_fingerprint = CaseObsName)
    

#     # save the new caseobs to the disk if necessary
#     if len(fn_caseobs_Phi.new_calculated_caseobs) > 0 and use_caseobs_from_disk == True: 
#         # save the new caseobs to the disk.
#         fn_caseobs_Phi.save_new_caseobs_to_ds_caseobs()
#         # save the vocab to the disk.
#         CaseObsFolder_vocab = fn_caseobs_Phi.CaseObsFolder_vocab
#         df_Vocab = pd.DataFrame({CaseObsName: CaseTknVocab})
#         df_Vocab.to_pickle(CaseObsFolder_vocab)
        
#     data_CaseObs_Results = {}
#     data_CaseObs_Results['ds_caseobs'] = ds_caseobs
#     data_CaseObs_Results['CaseObsName'] = CaseObsName
#     data_CaseObs_Results['RecObsName_to_RecObsInfo'] = RecObsName_to_RecObsInfo
#     data_CaseObs_Results['fn_caseobs_Phi'] = fn_caseobs_Phi
#     data_CaseObs_Results['CaseTknVocab'] = CaseTknVocab
#     return data_CaseObs_Results


# # ------------------------------------ pipeline for a case list ------------------------------------ #
# def pipeline_to_generate_co_to_CaseObsInfo(ds_case, 
#                                            co_to_CaseObsNameInfo, 
#                                            SPACE, 
#                                            cohort_args,
#                                            case_id_columns, 
#                                            record_to_ds_rec, 
#                                            record_to_ds_rec_info,
#                                            use_caseobs_from_disk,
#                                            batch_size = 1000):

#     co_to_CaseObsInfo = {}

#     for co, CaseObsNameInfo in co_to_CaseObsNameInfo.items():
#         CaseObsInfo = {}
#         Record_Observations_List = CaseObsNameInfo['Record_Observations_List']
#         CaseTkn = CaseObsNameInfo['CaseTkn']
        
#         logger.info(f'CaseObsNameInfo: {Record_Observations_List}, {CaseTkn}')

#         old_columns = list(ds_case.column_names)
#         # print('check whether ds_case will change along the way ----------->', ds_case)
#         data_CaseObs_Results = pipeline_caseset_to_caseobservation(ds_case, 
#                                                                    Record_Observations_List, 
#                                                                    CaseTkn, 
#                                                                    SPACE, 
#                                                                    cohort_args, 
#                                                                    record_to_ds_rec, 
#                                                                    record_to_ds_rec_info, 
#                                                                    use_caseobs_from_disk,
#                                                                    batch_size)

#         # select columns and update column names
#         ds_caseobs = data_CaseObs_Results['ds_caseobs']
#         CaseTknVocab = data_CaseObs_Results['CaseTknVocab']
#         RecObsName_to_RecObsInfo = data_CaseObs_Results['RecObsName_to_RecObsInfo']

#         new_columns = [i for i in ds_caseobs.column_names if i not in old_columns]
#         ds_caseobs = ds_caseobs.select_columns(case_id_columns + new_columns)
#         for column in new_columns:
#             ds_caseobs = ds_caseobs.rename_column(column, co + '_' + column)

#         # save information to CaseObsInfo
#         CaseObsInfo['ds_caseobs'] = ds_caseobs
#         CaseObsInfo['CaseObsName'] = data_CaseObs_Results['CaseObsName']
#         CaseObsInfo['vocab_caseobs'] = CaseTknVocab
#         CaseObsInfo['RecObsName_to_RecObsInfo'] = RecObsName_to_RecObsInfo
#         co_to_CaseObsInfo[co] = CaseObsInfo

#     return co_to_CaseObsInfo


# def get_complete_dataset(co_to_CaseObsInfo):
#     # 1. merge the datasets together 
#     ds = None 
#     for idx, co in enumerate(co_to_CaseObsInfo):
#         CaseObsInfo = co_to_CaseObsInfo[co]
#         ds_caseobs = CaseObsInfo['ds_caseobs']
#         if idx == 0:
#             ds = ds_caseobs 
#         else:
#             assert len(ds) == len(ds_caseobs)
#             columns = [i for i in ds_caseobs.column_names if i not in ds.column_names]
#             for column in columns:
#                 ds = ds.add_column(column, ds_caseobs[column]) 
#     # print(ds) 
#     return ds


# def pipeline_case(ds_case_dict,     # C
#                   case_observations,# CO 
#                   CaseTaskOp,       # Gamma
#                   case_id_columns,
#                   cohort_args, 
#                   record_to_ds_rec, 
#                   record_to_ds_rec_info,
#                   use_caseobs_from_disk,
#                   SPACE):

#     co_to_CaseObsName, co_to_CaseObsNameInfo = convert_case_observations_to_co_to_observation(case_observations)
#     PipelineInfo = get_RecNameList_and_FldTknList(co_to_CaseObsNameInfo)

#     d = {}
#     for case_type, ds_case in ds_case_dict.items():
#         co_to_CaseObsInfo = pipeline_to_generate_co_to_CaseObsInfo(ds_case, 
#                                                                     co_to_CaseObsNameInfo, 
#                                                                     SPACE, 
#                                                                     cohort_args,
#                                                                     case_id_columns, 
#                                                                     record_to_ds_rec, 
#                                                                     record_to_ds_rec_info,
#                                                                     use_caseobs_from_disk,
#                                                                     batch_size = 1000)
#         ds = get_complete_dataset(co_to_CaseObsInfo)
#         d[case_type] = ds

#     co_to_CaseObsVocab = {co: CaseObsInfo['vocab_caseobs'] for co, CaseObsInfo in co_to_CaseObsInfo.items()}
#     co_to_RecObsName_to_RecObsInfo = {co: CaseObsInfo['RecObsName_to_RecObsInfo'] for co, CaseObsInfo in co_to_CaseObsInfo.items()}
#     ds_caseobslist_dict = datasets.DatasetDict(d)

#     Gamma_tools = fetch_casetaskop_Gamma_tools(CaseTaskOp, SPACE)
#     get_case_op_vocab = Gamma_tools['get_case_op_vocab']
#     CaseTaskOpVocab = get_case_op_vocab(co_to_CaseObsVocab)
#     SeqTypeList = Gamma_tools.get('SeqTypeList', ['input_ids', 'labels'])
#     fn_Case_Operation_Tkn = Gamma_tools['fn_Case_Operation_Tkn']
#     ds_case_proc = ds_caseobslist_dict.map(lambda examples: fn_Case_Operation_Tkn(examples, co_to_CaseObsVocab, CaseTaskOpVocab), 
#                                            batch_size = Gamma_tools.get('batch_size', 1000), 
#                                            batched = Gamma_tools.get('BATCHED', 1))
    
    
#     results = {}
#     results['PipelineInfo'] = PipelineInfo
#     results['co_to_RecObsName_to_RecObsInfo'] = co_to_RecObsName_to_RecObsInfo
#     results['co_to_CaseObsName'] = co_to_CaseObsName
#     results['co_to_CaseObsNameInfo'] = co_to_CaseObsNameInfo
#     results['co_to_CaseObsVocab'] = co_to_CaseObsVocab
#     results['ds_caseobslist_dict'] = ds_caseobslist_dict
#     results['CaseTaskOpVocab'] = CaseTaskOpVocab
#     results['SeqTypeList'] = SeqTypeList
#     results['ds_case_proc'] = ds_case_proc

#     return results


# def create_tokenizer(CaseTaskOpVocab, tokenizer_folder):

#     tokenizer_dict = {}
#     for sequence_name in CaseTaskOpVocab:

#         tid2tkn = CaseTaskOpVocab[sequence_name]['tid2tkn']
#         tkn2tid = {v: k for k, v in tid2tkn.items()}
#         unk_token = tid2tkn[0]
#         if 'unk' not in unk_token:
#             print(f'Warning: unk_token: {unk_token} is not unk')

#         tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab = tkn2tid, unk_token=unk_token))
#         tokenizer.pre_tokenizer = WhitespaceSplit()

#         if not os.path.exists(tokenizer_folder):
#             # os.makedirs(os.path.dirname(tokenizer_folder))
#             os.makedirs(tokenizer_folder)

#         tokenizer_path = os.path.join(tokenizer_folder, sequence_name + '.json')
#         if os.path.exists(tokenizer_path):
#             tokenizer_old = tokenizers.Tokenizer.from_file(tokenizer_path)
#             old_dict = tokenizer_old.get_vocab()    
#             new_dict = tokenizer.get_vocab()
#             diff1 = old_dict.keys() - new_dict.keys()
#             diff2 = new_dict.keys() - old_dict.keys()
#             assert len(diff1) == 0 and len(diff2) == 0
#         else:
#             tokenizer.save(tokenizer_path)

#         tokenizer_dict[sequence_name] = tokenizer
#     return tokenizer_dict
