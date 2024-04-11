import os
import json
import yaml
import logging
import pickle 
import pandas as pd
import numpy as np
import datasets
from datasets import concatenate_datasets

import tokenizers
from tokenizers.pre_tokenizers import WhitespaceSplit


from .obsname import convert_case_observations_to_co_to_observation, get_RecNameList_and_FldTknList

from .loadtools import load_module_variables, load_ds_rec_and_info
from .loadtools import fetch_casetag_tools, fetch_casefilter_tools

from .loadtools import fetch_trigger_tools
from .pipeline_case import get_ds_case_to_process
from .pipeline_case import process_df_casefeat_tasks_in_chunks
from .pipeline_case import process_df_tagging_tasks_in_chunks
from .pipeline_case import process_df_filtering_tasks
from .pipeline_case import assign_caseSplitTag_to_dsCase
from .pipeline_case import get_dfset_from_SetName


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s:%(asctime)s:(%(filename)s@%(lineno)d %(name)s)]: %(message)s')


def pipeline_to_generate_dfcase_and_dataset(RecName_to_dsRec, 
                                            RecName_to_dsRecInfo,

                                            # df_case
                                            InputCaseSetName,
                                            TriggerCaseMethod,
                                            TagMethod_List,
                                            FilterMethod_List,
                                            SplitDict,
                                            
                                            # ds_case
                                            CaseSplitConfig,
                                            CaseFeat_List,

                                            # config
                                            cf_to_QueryCaseFeatConfig,
                                            cf_to_CaseFeatConfig, 
                                            SPACE,
                                            cohort_args, 
                                            cohort_label_list,
                                            
                                            # proc sets
                                            CASE_TAGGING_PROC_CONFIG,
                                            CASE_FIEDLING_PROC_CONFIG,
                                            
                                            SAVE_DF_CASE = False,
                                            SAVE_DS_DATA = False,
                                            
                                            LOAD_DF_CASE = False, 
                                            LOAD_DS_DATA = False, 

                                            RANDOM_SAMPLE = None,
                                            SAVE_TRIGGER_DF = True,
                                            ):
    
    results = {}
    
    # =========================== Part 0: df_case ===========================


    logger.info(f'-------------- (0) RecName_to_dsRec: {[i for i in RecName_to_dsRec]} --------------')
    # RecName_to_dsRec; RecName_to_dsRecInfo
    

    logger.info(f'-------------- (1) TriggerCaseMethod: {TriggerCaseMethod} --------------')
    Trigger_Tools = fetch_trigger_tools(TriggerCaseMethod, SPACE)
    case_id_columns = Trigger_Tools['case_id_columns']
    cohort_args['case_id_columns'] = case_id_columns


    logger.info(f'-------------- (2) InputCaseSetName: {InputCaseSetName} --------------')
    InputCaseSetName, df_case = get_ds_case_to_process(InputCaseSetName, 
                                                        cohort_label_list, 
                                                        TriggerCaseMethod, 
                                                        cohort_args, 
                                                        SPACE, 
                                                        RecName_to_dsRec, 
                                                        RecName_to_dsRecInfo, 
                                                        SAVE_TRIGGER_DF,
                                                        )
    logger.info(f'InputCaseSetName: {InputCaseSetName}')
    logger.info(f'df_case shape: {df_case.shape}')
    
    results['case_id_columns'] = case_id_columns

    
    # =========================== Part 1: df_case ===========================

    FinalOutCaseSetName = InputCaseSetName
    if len(TagMethod_List) > 0:
        FinalOutCaseSetName = FinalOutCaseSetName + '-' + '.'.join(TagMethod_List)
    if len(FilterMethod_List) > 0:
        FinalOutCaseSetName = FinalOutCaseSetName + '-' + '.'.join(FilterMethod_List)
    if len(SplitDict) > 0:
        SplitDict['RootID'] = cohort_args['RootID']
        SplitDict['ObsDT'] = cohort_args['ObsDTName']
        RANDOM_SEED = SplitDict['RANDOM_SEED']
        downsample_ratio = SplitDict['downsample_ratio']
        out_ratio = SplitDict['out_ratio']
        test_ratio = SplitDict['test_ratio']
        valid_ratio = SplitDict['valid_ratio']
        SplitMethod = f'rs{RANDOM_SEED}.ds{downsample_ratio}.out{out_ratio}ts{test_ratio}vd{valid_ratio}' 
        FinalOutCaseSetName = FinalOutCaseSetName + '-' + SplitMethod

    path = os.path.join(SPACE['DATA_CaseSet'], FinalOutCaseSetName + '.p')
    # print(os.getcwd())
    # print(path)
    # print(os.path.exists(path))
    # return None 
    logger.info(f'ds_case path: {path}')
    if os.path.exists(path) and LOAD_DF_CASE == True:
        logger.info(f'-------------- load df_case from DATA_CASE --------------')
        logger.info(f'load df_case from: {path}')
        df_case = pd.read_pickle(path)
        PROC_DF_CASE = False
    else: 
        logger.info(f'-------------- execute processing df_case --------------')
        PROC_DF_CASE = True



    if PROC_DF_CASE: 
        if RANDOM_SAMPLE is not None:
            df_case = df_case.sample(RANDOM_SAMPLE, random_state = 42).reset_index(drop = True)
            logger.info(f'randomly selected df_case shape: {df_case.shape}')



        logger.info(f'-------------- (1.a) TagMethod_List: {TagMethod_List} --------------')
        if len(TagMethod_List) > 0:
            OutputCaseSetName, df_case = process_df_tagging_tasks_in_chunks(df_case, cohort_label_list, case_id_columns, 
                                                                            InputCaseSetName, 
                                                                            TagMethod_List, cf_to_QueryCaseFeatConfig, 
                                                                            cohort_args, SPACE, 
                                                                            RecName_to_dsRec, RecName_to_dsRecInfo,
                                                                            **CASE_TAGGING_PROC_CONFIG)
            logger.info(f'df_case shape: {df_case.shape}')
            


        logger.info(f'-------------- (1.b) FilterMethod_List: {FilterMethod_List} --------------')
        if len(FilterMethod_List) > 0:
            # logger.info(f'---------- before filtering: {df_case.shape} --------------')
            df_case = process_df_filtering_tasks(df_case, FilterMethod_List, SPACE)
            # logger.info(f'---------- after filtering: {df_case.shape} --------------')
            logger.info(f'df_case shape: {df_case.shape}')
            

        logger.info(f'-------------- (1.c) SplitDict: {SplitDict} --------------')
        if len(SplitDict) > 0:
            SplitDict['RootID'] = cohort_args['RootID']
            SplitDict['ObsDT'] = cohort_args['ObsDTName']
            df_case = assign_caseSplitTag_to_dsCase(df_case,  **SplitDict)


    if SAVE_DF_CASE == True:
        logger.info(f'-------------- SAVE df_case to DATA_CASE --------------')
        path = os.path.join(SPACE['DATA_CaseSet'], FinalOutCaseSetName + '.p')
        logger.info(f'save df_case to: {path}')
        df_case.to_pickle(path)


    results['FinalOutCaseSetName'] = FinalOutCaseSetName
    results['df_case'] = df_case.copy()


    # =========================== Part 2: ds_data ===========================
    logger.info(f'-------------- SAVE df_case to DATA_CASE --------------')
    dataset_name = '.'.join(CaseFeat_List)  
    dataset_path = os.path.join(SPACE['DATA_TASK'], FinalOutCaseSetName, dataset_name)

    # print(os.getcwd())
    # print(dataset_path)
    # print(os.path.exists(dataset_path))
    # return None

    if os.path.exists(dataset_path) and LOAD_DS_DATA == True:
        logger.info(f'-------------- load ds_case from DATA_TASK --------------')
        
        data_path = dataset_path + '/data'
        logger.info(f'load ds_case from: {data_path}')
        ds_case_cf_dict = datasets.DatasetDict.load_from_disk(data_path)

        vocab_path = dataset_path + '/vocab'
        with open(vocab_path, 'rb') as f:
            cf_to_CFVocab = pickle.load(f)

        PROC_DS_DATA = False

    else:
        logger.info(f'-------------- execute processing ds_case --------------')
        PROC_DS_DATA = True



    if PROC_DS_DATA:
        logger.info(f'-------------- (2.a) Dataset Split: {CaseSplitConfig} --------------')
        ds_case_dict = {}

        case_id_columns = Trigger_Tools['case_id_columns']

        if len(CaseSplitConfig) > 0:
            df_case_learning = df_case.copy()
            TrainSetName = CaseSplitConfig['TrainSetName']
            EvalSetNames = CaseSplitConfig['EvalSetNames']
            
            SubGroupFilterMethod = {}
            d = {}
            for SetName in [TrainSetName] + EvalSetNames:
                df_set = get_dfset_from_SetName(df_case_learning, SetName, case_id_columns, SubGroupFilterMethod)
                ds_set = datasets.Dataset.from_pandas(df_set)
                d[SetName] = ds_set
            ds_case_dict = datasets.DatasetDict(d)

        else:
            d = {'inference': datasets.Dataset.from_pandas(df_case[case_id_columns].reset_index(drop = True))}
            ds_case_dict = datasets.DatasetDict(d)
        
        logger.info(f'ds_case_dict: {ds_case_dict}')
    


        logger.info(f'-------------- (2.b) Dataset Case Fielding: {CaseFeat_List} --------------')
        ds_case_cf_dict = {}
        cf_to_CFVocab = {}
        if len(CaseFeat_List) > 0:
            for split, ds in ds_case_dict.items():
                df_case_neat = ds.to_pandas()
                cf_to_CaseFeatInfo, df_case_neat = process_df_casefeat_tasks_in_chunks(df_case_neat, 
                                                                                cohort_label_list,
                                                                                case_id_columns,
                                                                                InputCaseSetName, 
                                                                                CaseFeat_List, 
                                                                                cf_to_CaseFeatConfig, 
                                                                                cohort_args,
                                                                                SPACE, 
                                                                                RecName_to_dsRec, 
                                                                                RecName_to_dsRecInfo,
                                                                                **CASE_FIEDLING_PROC_CONFIG)
                
                ds_case_cf_dict[split] = df_case_neat
            cf_to_CFVocab = {cf: Info['CF_vocab'] for cf, Info in cf_to_CaseFeatInfo.items()}


    if SAVE_DS_DATA == True and len(ds_case_cf_dict) > 0:
        logger.info(f'-------------- SAVE df_case to DATA_CASE --------------')
        dataset_name = '.'.join(CaseFeat_List)  
        dataset_path = os.path.join(SPACE['DATA_TASK'], InputCaseSetName, dataset_name)
        
        # dataset
        data_path = dataset_path + '/data'
        logger.info(f'data_path: {data_path}')
        ds_case_cf_dict = datasets.DatasetDict(ds_case_cf_dict) 
        ds_case_cf_dict.save_to_disk(data_path)
        
        # vocab
        vocab_path = dataset_path + '/vocab'
        logger.info(f'vocab_path: {vocab_path}')
        with open(vocab_path, 'wb') as f:
            pickle.dump(cf_to_CFVocab, f)

    results['cf_to_CFVocab'] = cf_to_CFVocab
    results['ds_case_cf_dict'] = ds_case_cf_dict
    logger.info(ds_case_cf_dict)
    logger.info(f'-------------- Done --------------')  
    return results










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


# def get_dfset_from_SetName(df_dsmp, SetName, case_id_columns, SubGroupFilterMethod):
#     df_dsmp = df_dsmp.copy()
#     Split, SubGroup = SetName.split('_')
#     for i in ['In', 'Out', 'Train', 'Valid', 'Test']:
#         if i.lower() in [t.lower() for t in Split.split('-')]:
#             df_dsmp = df_dsmp[df_dsmp[i]].reset_index(drop = True)
#     df_set = df_dsmp[case_id_columns].reset_index(drop = True)
#     if SubGroup.lower() != 'all': pass
#     return df_set



# def get_tkn_value(x, tkn_id, ids, wgts):
#     if tkn_id in x[ids]:
#         index = list(x[ids]).index(tkn_id)
#         return x[wgts][index]
#     else:
#         return 0
            

# def load_complete_PipelineInfo(datapoint_args, base_config, use_inference):

#     case_observations_total = []
#     for k, v in datapoint_args.items():
#         if use_inference == True and 'Output' in k: continue 
#         case_observations_total = case_observations_total + v['case_observations']

#     _, co_to_CaseObsNameInfo = convert_case_observations_to_co_to_observation(case_observations_total)
#     PipelineInfo = get_RecNameList_and_FldTknList(co_to_CaseObsNameInfo)
#     PipelineInfo['FldTknList'] = [i+'Tkn' for i in PipelineInfo['FldTknList']]

#     # 3. get record_sequence
#     record_sequence = PipelineInfo['RecNameList']
#     RecName_to_PrtRecName = base_config['RecName_to_PrtRecName']
#     record_sequence_prt = [RecName_to_PrtRecName[recname] for recname in record_sequence if RecName_to_PrtRecName[recname] != 'None']
#     # print(record_sequence_prt)
#     new_records = [i for i in record_sequence_prt if i not in record_sequence]
#     while len(new_records) > 0:
#         record_sequence.extend(new_records)
#         record_sequence_prt = [RecName_to_PrtRecName[recname] for recname in record_sequence if RecName_to_PrtRecName[recname] != 'None']
#         new_records = [i for i in record_sequence_prt if i not in record_sequence]
#     record_sequence = [recname for recname in base_config['RecName_Sequence'] if recname in PipelineInfo['RecNameList']]

#     PipelineInfo['RecNameList'] = record_sequence # 
#     return PipelineInfo


# def get_ds_case_to_process(InputCaseSetName, 
#                            cohort_label_list, 
#                            TriggerCaseMethod, 
#                            base_config, 
#                            SPACE,
#                            RecName_to_dsRec = {},
#                            RecName_to_dsRecInfo = {}):
#     '''
#         this function is designed to be used in the training pipeline only.
#         We only want to get a ds_case to tag, filter, split (optional), training, and inference. 
#     '''
#     if InputCaseSetName is None: 
#         COHORT = 'C' + '.'.join([str(i) for i in cohort_label_list])
#         TRIGGER = TriggerCaseMethod
#         InputCaseSetName = '-'.join([COHORT, TRIGGER])
        
#     InputCaseFolder = os.path.join(SPACE['DATA_CaseSet'], InputCaseSetName)
#     InputCaseFile = InputCaseFolder + '.p'

        
#     if os.path.exists(InputCaseFolder):
#         assert os.path.isdir(InputCaseFolder)
#         L = []
#         if 'TaggingSize' in InputCaseFolder:
#             tag_method_list = sorted(os.listdir(InputCaseFolder))
#             for tag_method in tag_method_list:
#                 file_list = sorted(os.listdir(os.path.join(InputCaseFolder, tag_method)))
#                 df_case_tag = pd.concat([pd.read_pickle(os.path.join(InputCaseFolder, tag_method, f)) 
#                                         for f in file_list])
#                 df_case_tag = df_case_tag.reset_index(drop = True)
#                 L.append(df_case_tag)
#             pypath = os.path.join(base_config['trigger_pyfolder'], f'{TriggerCaseMethod}.py')
#             module = load_module_variables(pypath)
#             case_id_columns = module.case_id_columns
#             df_case = L[0]
#             for df_case_tag in L[1:]:
#                 assert len(df_case) == len(df_case_tag)
#                 columns = [i for i in df_case_tag.columns if i not in df_case.columns]
#                 df_case = pd.merge(df_case, df_case_tag[case_id_columns + columns], on=case_id_columns)
#         else:
#             file_list = sorted(os.listdir(InputCaseFolder))
#             df_case = pd.concat([pd.read_pickle(os.path.join(InputCaseFolder, f)) for f in file_list])

#     elif os.path.exists(InputCaseFile):
#         assert os.path.exists(InputCaseFile)
#         df_case = pd.read_pickle(InputCaseFile)
#     else: 
#         ################## get df_case to tag ##################
#         # base_config = load_cohort_args(recfldtkn_config_path, SPACE)
#         Trigger_Tools = fetch_trigger_tools(TriggerCaseMethod, SPACE)
#         case_id_columns = Trigger_Tools['case_id_columns']
#         special_columns = Trigger_Tools['special_columns'] 
#         TriggerRecName = Trigger_Tools['TriggerRecName']
#         convert_TriggerEvent_to_Caseset = Trigger_Tools['convert_TriggerEvent_to_Caseset']
        
#         if TriggerRecName in RecName_to_dsRec:
#             ds_rec = RecName_to_dsRec[TriggerRecName]
#         else:
#             ds_rec, _ = load_ds_rec_and_info(TriggerRecName, base_config, cohort_label_list)
        
#         # ds_rec, _ = load_ds_rec_and_info(TriggerRecName, base_config, cohort_label_list)
#         df_case = convert_TriggerEvent_to_Caseset(ds_rec, case_id_columns, special_columns, base_config)
#         InputCaseFile = InputCaseFolder + '.p'
#         df_case.to_pickle(InputCaseFile)
#     # print(df_case.shape)
#     return InputCaseSetName, df_case
    

# def fn_casefeat_querying(ds_case, 
#                          base_config, 
#                          case_id_columns,
#                          case_observations, 
#                          name_CaseGamma, 
#                          RecName_to_dsRec = {},
#                          RecName_to_dsRecInfo = {}, 
#                          use_CF_from_disk = True, 
#                          use_CO_from_disk = True):
    
#     #############################
#     Gamma_Config = {
#         'case_observations':case_observations,
#         'name_CaseGamma': name_CaseGamma, # CF
#     }
#     #############################

#     # case_id_columns = base_config['case_id_columns']
#     df_case = ds_case.select_columns(case_id_columns).to_pandas()
#     CaseFeatInfo = get_fn_case_GammaFullInfo(Gamma_Config, 
#                                             base_config, 
#                                             RecName_to_dsRec, 
#                                             RecName_to_dsRecInfo, 
#                                             df_case,
#                                             use_CF_from_disk,
#                                             use_CO_from_disk)
    
#     FnCaseFeatGamma = CaseFeatInfo['FnCaseFeatGamma']
#     batch_size = CaseFeatInfo.get('batch_size', 1000)
#     CaseFeatName = CaseFeatInfo['CaseFeatName']
#     # CF_vocab = CaseFeatInfo['CF_vocab']
#     ds_case = ds_case.map(FnCaseFeatGamma, 
#                             batched = True, 
#                             batch_size= batch_size, 
#                             load_from_cache_file = False, 
#                             new_fingerprint = CaseFeatName)
    
#     if len(FnCaseFeatGamma.new_CFs) > 0 and use_CF_from_disk == True:
#         logger.info(f'----- Save CF {CaseFeatName}: to Cache File -----')
#         FnCaseFeatGamma.save_new_CFs_to_disk()
    
#     for COName, FnCaseObsPhi in FnCaseFeatGamma.COName_to_FnCaseObsPhi.items():
#         if len(FnCaseObsPhi.new_COs) > 0 and use_CO_from_disk == True:
#             logger.info(f'----- Save CO {COName}: to Cache File -----')
#             FnCaseObsPhi.save_new_COs_to_disk()

#     return CaseFeatInfo, ds_case


# ######### tagging tasks #########
# def process_df_tagging_tasks(df_case, 
#                              cohort_label_list,
#                              case_id_columns,
#                              InputCaseSetName, 
#                              TagMethod_List, 
#                              cf_to_QueryCaseFeatConfig,  
#                              base_config,
#                              SPACE, 
#                              RecName_to_dsRec, 
#                              RecName_to_dsRecInfo,
#                              use_CF_from_disk, 
#                              use_CO_from_disk, 
#                              chunk_id = None, 
#                              start_idx = None, 
#                              end_idx = None, 
#                              chunk_size = 500000,
#                              save_to_pickle = False,
#                              ):
    
#     # You can also check whether it is in disk or not.
#     OutputCaseSetName = '-'.join([InputCaseSetName, 't.'+'.'.join(TagMethod_List)])
    
#     for TagMethod in TagMethod_List:
#         # logger.info(f'--------- TagMethod {TagMethod} -------------')
#         # logger.info(f'--------- before tagging {df_case.shape} -------------')
#         if TagMethod in cf_to_QueryCaseFeatConfig:
#             QueryCaseFeatConfig = cf_to_QueryCaseFeatConfig[TagMethod]
#             TagMethodFile = TagMethod + '.' + Hasher.hash(QueryCaseFeatConfig)
#         else:
#             TagMethodFile = TagMethod

#         ############################### option 1: loading from disk ###############################
#         if save_to_pickle == True:
#             Folder = os.path.join(SPACE['DATA_CaseSet'], InputCaseSetName + '-Tagging' + f'Size{chunk_size}')
#             if chunk_id == None: chunk_id = 0
#             if start_idx == None: start_idx = 0
#             if end_idx == None: end_idx = len(df_case)

#             start_idx_k = start_idx // 1000
#             end_idx_k = end_idx // 1000
#             filename = f'idx{chunk_id:05}_{start_idx_k:06}k_{end_idx_k:06}k.p'
#             fullfilepath = os.path.join(Folder, TagMethodFile, filename)
#             fullfolderpath = os.path.dirname(fullfilepath)

#             if not os.path.exists(fullfolderpath): 
#                 os.makedirs(fullfolderpath)


#             # print(fullfilepath)
#             # print(os.path.exists(fullfilepath))
#             if os.path.exists(fullfilepath): 
#                 df_case_new = pd.read_pickle(fullfilepath) 
#                 columns = [i for i in df_case_new.columns if i not in df_case.columns]
#                 df_case = pd.merge(df_case, df_case_new[case_id_columns + columns], on=case_id_columns)
#                 continue 

#         ############################### option 2: calculating and saving to disk ###############################
#         if TagMethod in cf_to_QueryCaseFeatConfig:
#             QueryCaseFeatConfig = cf_to_QueryCaseFeatConfig[TagMethod]
#             case_observations = QueryCaseFeatConfig['case_observations']
#             name_CaseGamma = QueryCaseFeatConfig['name_CaseGamma']
#             tkn_name_list = QueryCaseFeatConfig['tkn_name_list']
            
#             ds_case = datasets.Dataset.from_pandas(df_case)
#             CaseFeatInfo, ds_case = fn_casefeat_querying(ds_case, 
#                                                          base_config, 
#                                                          case_id_columns,
#                                                          case_observations, 
#                                                          name_CaseGamma, 
#                                                          RecName_to_dsRec,
#                                                          RecName_to_dsRecInfo, 
#                                                          use_CF_from_disk, 
#                                                          use_CO_from_disk)
            
#             # assert name_CaseGamma is a special CaseGamma
#             CF_vocab = CaseFeatInfo['CF_vocab']
#             CaseFeatName = CaseFeatInfo['CaseFeatName']
#             rename_dict = {i: CaseFeatName + ':' + i for i in CF_vocab}
#             df_case = ds_case.to_pandas().rename(columns=rename_dict)

            
#             ids  = CaseFeatName + ':' + 'input_ids'
#             wgts = CaseFeatName + ':' + 'input_wgts'
#             for tkn_name in tkn_name_list:
#                 tkn_id = CF_vocab['input_ids']['tkn2tid'][tkn_name]
#                 df_case[tkn_name] = df_case.apply(lambda x: get_tkn_value(x, tkn_id, ids, wgts), axis = 1)
#             df_case = df_case.drop(columns = [ids, wgts])
            
#         else:
#             pypath = os.path.join(SPACE['CODE_FN'], 'fn_learning', f'{TagMethod}.py')
#             module = load_module_variables(pypath)
#             MetaDict = module.MetaDict
#             if 'InfoRecName' in MetaDict:
#                 InfoRecName, subgroup_columns, fn_case_tagging = module.InfoRecName, module.subgroup_columns, module.fn_case_tagging
#                 ds_info, _ = load_ds_rec_and_info(InfoRecName, base_config, cohort_label_list)
#                 df_case = fn_case_tagging(df_case, ds_info, subgroup_columns, base_config)
#             elif 'fn_case_tagging_on_casefeat' in MetaDict:
#                 fn_case_tagging_on_casefeat = MetaDict['fn_case_tagging_on_casefeat']
#                 df_case = fn_case_tagging_on_casefeat(df_case)
#             else:
#                 raise ValueError('No fn_case_tagging_on_casefeat or InfoRecName in the module')
            
#         if save_to_pickle == True:
#             df_case.to_pickle(fullfilepath)

#     return OutputCaseSetName, df_case
    

# def _process_chunk_tagging(chunk_id, df_case, chunk_size, 
#                            cohort_label_list, case_id_columns,
#                            InputCaseSetName, TagMethod_List, cf_to_QueryCaseFeatConfig, base_config,
#                            SPACE, RecName_to_dsRec, RecName_to_dsRecInfo, 
#                            use_CF_from_disk, use_CO_from_disk, save_to_pickle):
#     start_idx = chunk_id * chunk_size
#     end_idx = min((chunk_id + 1) * chunk_size, len(df_case))
#     df_case_chunk = df_case.iloc[start_idx:end_idx].reset_index(drop=True)
#     _, df_case_chunk_tagged = process_df_tagging_tasks(
#         df_case_chunk, 
#         cohort_label_list,
#         case_id_columns,
#         InputCaseSetName, 
#         TagMethod_List, 
#         cf_to_QueryCaseFeatConfig,  
#         base_config,
#         SPACE, 
#         RecName_to_dsRec, 
#         RecName_to_dsRecInfo,
#         use_CF_from_disk, 
#         use_CO_from_disk,
#         chunk_id, 
#         start_idx, 
#         end_idx, 
#         chunk_size,
#         save_to_pickle,
#     )
#     return df_case_chunk_tagged


# def process_df_tagging_tasks_in_chunks(df_case, 
#                                        cohort_label_list,
#                                        case_id_columns,
#                                        InputCaseSetName, 
#                                        TagMethod_List,
#                                        cf_to_QueryCaseFeatConfig,
#                                        base_config, 
#                                        SPACE, 
#                                        RecName_to_dsRec, 
#                                        RecName_to_dsRecInfo,
#                                        use_CF_from_disk,
#                                        use_CO_from_disk,
#                                        start_chunk_id = 0, 
#                                        end_chunk_id = None, 
#                                        chunk_size = 500000,
#                                        save_to_pickle = True, 
#                                        num_processors = 0):
    
#     # --------------- get the tagging method name ---------------
#     OutputCaseSetName = '-'.join([InputCaseSetName, 'Tagging'])
#     Folder = os.path.join(SPACE['DATA_CaseSet'], OutputCaseSetName + f'Size{chunk_size}')
    
#     # --------------- process tagging tasks ---------------
#     if end_chunk_id is None: end_chunk_id = len(df_case) // chunk_size + 1
#     chunk_id_list = range(start_chunk_id, end_chunk_id)



#     df_case_chunk_tagged_list = []
#     if num_processors > 1:
#         # with ProcessPoolExecutor(max_workers=num_processors) as executor:
#         #     df_case_chunk_tagged_list = list(executor.map(process_chunk, chunk_id_list))
#         with ProcessPoolExecutor(max_workers=num_processors) as executor:
#             futures = [executor.submit(
#                         _process_chunk_tagging, 
#                             chunk_id, df_case, chunk_size, 
#                             cohort_label_list, case_id_columns,
#                             InputCaseSetName, TagMethod_List, cf_to_QueryCaseFeatConfig, base_config,
#                             SPACE, RecName_to_dsRec, RecName_to_dsRecInfo, 
#                             use_CF_from_disk, use_CO_from_disk, save_to_pickle) 
                
#                 for chunk_id in chunk_id_list]

#             for future in as_completed(futures):
#                 df_case_chunk_tagged_list.append(future.result())

#     else:
#         for chunk_id in chunk_id_list:
#             df_case_chunk_tagged = _process_chunk_tagging(
#                                         chunk_id, df_case, chunk_size, 
#                                         cohort_label_list, case_id_columns,
#                                         InputCaseSetName, TagMethod_List, cf_to_QueryCaseFeatConfig, base_config,
#                                         SPACE, RecName_to_dsRec, RecName_to_dsRecInfo, 
#                                         use_CF_from_disk, use_CO_from_disk, save_to_pickle)
#             df_case_chunk_tagged_list.append(df_case_chunk_tagged)

#     df_case_tagged_final = pd.concat(df_case_chunk_tagged_list, axis = 0).reset_index(drop = True)
#     OutputCaseSetName_final = '-'.join([InputCaseSetName, 't.'+'.'.join(TagMethod_List)])
#     return OutputCaseSetName_final, df_case_tagged_final


# ######### casefeat tasks #########
# def process_df_casefeat_tasks(df_case, 
#                               cohort_label_list,
#                               case_id_columns,
#                               InputCaseSetName, 
#                               CaseFeat_List, 
#                               cf_to_CaseFeatConfig, 
#                               base_config,
#                               SPACE, 
#                               RecName_to_dsRec, 
#                               RecName_to_dsRecInfo,
#                               use_CF_from_disk, 
#                               use_CO_from_disk, 
#                               chunk_id, 
#                               start_idx, 
#                               end_idx, 
#                               chunk_size,
#                               save_to_pickle):
    
#     ds_case = datasets.Dataset.from_pandas(df_case)
#     cf_to_CaseFeatInfo = {}
#     for CaseFeat in CaseFeat_List:
#         logger.info(f'--------- CaseFeat {CaseFeat} -------------')

#         assert  CaseFeat in cf_to_CaseFeatConfig
#         CaseFeatConfig = cf_to_CaseFeatConfig[CaseFeat]
#         case_observations = CaseFeatConfig['case_observations']
#         name_CaseGamma = CaseFeatConfig['name_CaseGamma']
#         old_columns = ds_case.column_names
#         CaseFeatInfo, ds_case = fn_casefeat_querying(ds_case, 
#                                                      base_config, 
#                                                      case_id_columns,
#                                                      case_observations, 
#                                                      name_CaseGamma, 
#                                                      RecName_to_dsRec,
#                                                      RecName_to_dsRecInfo, 
#                                                      use_CF_from_disk, 
#                                                      use_CO_from_disk)
        
#         new_columns = [i for i in ds_case.column_names if i not in old_columns]
#         rename_dict = {i: CaseFeat + '.' + i for i in new_columns}
#         for old_name, new_name in rename_dict.items():
#             ds_case = ds_case.rename_column(old_name, new_name)
#         cf_to_CaseFeatInfo[CaseFeat] = CaseFeatInfo
#     # OutputCaseSetName = '-'.join([InputCaseSetName, 'cf.'+'.'.join(CaseFeat_List)])
#     return cf_to_CaseFeatInfo, ds_case


# def _process_chunk_casefeat(chunk_id, df_case, chunk_size, 
#                            cohort_label_list, case_id_columns,
#                            InputCaseSetName, CaseFeat_List, cf_to_CaseFeatConfig, base_config,
#                            SPACE, RecName_to_dsRec, RecName_to_dsRecInfo, 
#                            use_CF_from_disk, use_CO_from_disk, save_to_pickle):
#     start_idx = chunk_id * chunk_size
#     end_idx = min((chunk_id + 1) * chunk_size, len(df_case))
#     df_case_chunk = df_case.iloc[start_idx:end_idx].reset_index(drop=True)
#     _, df_case_chunk_casefeat = process_df_casefeat_tasks(
#         df_case_chunk, 
#         cohort_label_list,
#         case_id_columns,
#         InputCaseSetName, 
#         CaseFeat_List, 
#         cf_to_CaseFeatConfig,  
#         base_config,
#         SPACE, 
#         RecName_to_dsRec, 
#         RecName_to_dsRecInfo,
#         use_CF_from_disk, 
#         use_CO_from_disk,
#         chunk_id, 
#         start_idx, 
#         end_idx, 
#         chunk_size,
#         save_to_pickle,
#     )
#     return df_case_chunk_casefeat


# def process_df_casefeat_tasks_in_chunks(df_case, 
#                                         cohort_label_list,
#                                         case_id_columns,
#                                         InputCaseSetName, 
#                                         CaseFeat_List, 
#                                         cf_to_CaseFeatConfig, 
#                                         base_config,
#                                         SPACE, 
#                                         RecName_to_dsRec, 
#                                         RecName_to_dsRecInfo,
#                                         use_CF_from_disk, 
#                                         use_CO_from_disk,
#                                         start_chunk_id = 0, 
#                                         end_chunk_id = None, 
#                                         chunk_size = 500000,
#                                         save_to_pickle = False, 
#                                         num_processors = 1):
    
#     # --------------- cf_to_CaseFeatInfo ---------------
#     cf_to_CaseFeatInfo = {}
#     for CaseFeat in CaseFeat_List:
#         assert  CaseFeat in cf_to_CaseFeatConfig
#         CaseFeatConfig = cf_to_CaseFeatConfig[CaseFeat]
#         case_observations = CaseFeatConfig['case_observations']
#         name_CaseGamma = CaseFeatConfig['name_CaseGamma']
#         #############################
#         Gamma_Config = {
#             'case_observations':case_observations,
#             'name_CaseGamma': name_CaseGamma, # CF
#         }
#         #############################
#         # df_case = ds_case.select_columns(case_id_columns).to_pandas()
#         CaseFeatInfo = get_fn_case_GammaFullInfo(Gamma_Config, 
#                                                  base_config, 
#                                                  RecName_to_dsRec, 
#                                                  RecName_to_dsRecInfo, 
#                                                  df_case[case_id_columns],
#                                                  use_CF_from_disk,
#                                                  use_CO_from_disk)
#         cf_to_CaseFeatInfo[CaseFeat] = CaseFeatInfo
        
    
#     # --------------- get the tagging method name ---------------
#     OutputCaseSetName = '-'.join([InputCaseSetName, 'Tagging'])
#     Folder = os.path.join(SPACE['DATA_CaseSet'], OutputCaseSetName + f'Size{chunk_size}')

#     # --------------- process tagging tasks ---------------
#     if end_chunk_id is None: end_chunk_id = len(df_case) // chunk_size + 1
#     chunk_id_list = range(start_chunk_id, end_chunk_id)

#     ds_case_chunk_casefeat_list = []
#     if num_processors > 1:
#         # with ProcessPoolExecutor(max_workers=num_processors) as executor:
#         #     df_case_chunk_tagged_list = list(executor.map(process_chunk, chunk_id_list))
#         with ProcessPoolExecutor(max_workers = num_processors) as executor:
#             futures = [executor.submit(
#                         _process_chunk_casefeat, 
#                             chunk_id, df_case, chunk_size, 
#                            cohort_label_list, case_id_columns,
#                            InputCaseSetName, CaseFeat_List, cf_to_CaseFeatConfig, base_config,
#                            SPACE, RecName_to_dsRec, RecName_to_dsRecInfo, 
#                            use_CF_from_disk, use_CO_from_disk, save_to_pickle) 
                
#                 for chunk_id in chunk_id_list]

#             for future in as_completed(futures):
#                 ds_case_chunk_casefeat_list.append(future.result())
#                 # future.result()
#     else:
#         for chunk_id in chunk_id_list:
#             ds_case_chunk_casefeat = _process_chunk_casefeat(
#                                         chunk_id, df_case, chunk_size, 
#                                         cohort_label_list, case_id_columns,
#                                         InputCaseSetName, CaseFeat_List, cf_to_CaseFeatConfig, base_config,
#                                         SPACE, RecName_to_dsRec, RecName_to_dsRecInfo, 
#                                         use_CF_from_disk, use_CO_from_disk, save_to_pickle
#                                     ) 
#             ds_case_chunk_casefeat_list.append(ds_case_chunk_casefeat)
                            
#     ds_case = concatenate_datasets(ds_case_chunk_casefeat_list)
#     return cf_to_CaseFeatInfo, ds_case


# ######### split tasks #########
# def generate_random_tags(df_case, RANDOM_SEED, RootID, ObsDT):
#     np.random.seed(RANDOM_SEED)
#     # RootID, ObsDT = 'PID', 'ObsDT'

#     # down sample 
#     df_case['RandDownSample'] = np.random.rand(len(df_case))

#     # in&out
#     df_P = df_case[[RootID]].drop_duplicates().reset_index(drop = True)
#     df_P['RandInOut'] = np.random.rand(len(df_P))
#     df_case = pd.merge(df_case, df_P)

#     # test
#     df_case['CaseLocInP'] = df_case.groupby(RootID).cumcount()
#     df_case = pd.merge(df_case, df_case[RootID].value_counts().reset_index())
#     df_case['CaseRltLocInP'] = df_case['CaseLocInP'] /  df_case['count']
    
#     # test other options
#     df_case['RandTest'] = np.random.rand(len(df_case))

#     # validation
#     df_case['RandValidation'] = np.random.rand(len(df_case))

#     df_case = df_case.drop(columns = ['CaseLocInP', 'count']).reset_index(drop = True)
#     df_case = df_case.sort_values('RandDownSample').reset_index(drop = True)

#     random_columns = ['RandDownSample', 'RandInOut', 'CaseRltLocInP', 'RandTest', 'RandValidation']
#     return df_case, random_columns


# def assign_caseSplitTag_to_dsCase(df_case, 
#                                 RANDOM_SEED, 
#                                 RootID, 
#                                 ObsDT,
#                                 downsample_ratio, 
#                                 out_ratio, 
#                                 test_ratio, 
#                                 valid_ratio):

#     df = df_case 
#     df_rs, random_columns = generate_random_tags(df, RANDOM_SEED, RootID, ObsDT,)
#     df_dsmp = df_rs[df_rs['RandDownSample'] <= downsample_ratio].reset_index(drop = True)

#     df_dsmp['Out'] = df_dsmp['RandInOut'] < out_ratio
#     df_dsmp['In'] = df_dsmp['RandInOut'] >= out_ratio
#     assert df_dsmp[['Out', 'In']].sum(axis = 1).mean() == 1

#     if 'tail' in str(test_ratio):
#         TestSelector = 'CaseRltLocInP'
#         test_ratio = float(test_ratio.replace('tail', ''))
#         test_threshold = 1 - test_ratio
#     elif type(test_ratio) != float and type(test_ratio) != int:
#         TestSelector = 'ObsDT'
#         test_threshold = pd.to_datetime(test_ratio)
#     else:
#         TestSelector = 'RandTest'
#         test_threshold = 1 - test_ratio

#     if 'tail' in str(valid_ratio):
#         ValidSelector = 'CaseRltLocInP'
#         valid_ratio = float(valid_ratio.replace('tail', ''))
#         valid_threshold = 1 - valid_ratio
#     elif type(valid_ratio) != float and type(valid_ratio) != int:
#         ValidSelector = 'ObsDT'
#         valid_threshold = pd.to_datetime(valid_ratio)
#     else:
#         ValidSelector = 'RandTest' 
#         valid_threshold = 1 - valid_ratio
        
#     df_dsmp['Test'] = df_dsmp[TestSelector] > test_threshold
#     df_dsmp['Valid'] = (df_dsmp[ValidSelector] > valid_threshold) & (df_dsmp['Test'] == False)
#     df_dsmp['Train'] = (df_dsmp['Test'] == False) & (df_dsmp['Valid'] == False)

#     assert df_dsmp[['Train', 'Valid', 'Test']].sum(axis = 1).mean() == 1

#     df_dsmp = df_dsmp.drop(columns = random_columns)
#     return df_dsmp



# ######### filtering tasks #########
# def process_df_filtering_tasks(df_case, FilterMethod_List, SPACE):
#     for FilterMethod in FilterMethod_List:
#         logger.info(f'FilterMethod: {FilterMethod}')
#         pypath = os.path.join(SPACE['CODE_FN'], 'fn_learning', f'{FilterMethod}.py')
#         module = load_module_variables(pypath)

#         fn_case_filtering = module.fn_case_filtering
#         logger.info(f'before filtering: {df_case.shape}')
#         df_case = fn_case_filtering(df_case)
#         logger.info(f'after filtering: {df_case.shape}')
#     return df_case

# ######### inference tasks #########
