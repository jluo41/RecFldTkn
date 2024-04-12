import os
import time 
import shutil
import logging 
import numpy as np 
import pandas as pd
from typing import List
from datetime import datetime 

import datasets
from datasets import concatenate_datasets

from .configfn import load_record_args, load_fldtkn_args
from .loadtools import load_module_variables, load_ds_rec_and_info
from .loadtools import find_timelist_index, add_key_return_dict
from .loadtools import fetch_caseobs_Phi_tools, fetch_casefeat_Gamma_tools
from .obsname import parse_RecObsName, parse_CaseObsName, parse_CaseFeatName
from .obsname import convert_RONameList_to_COName, convert_CONameList_to_CFName
from .obsname import convert_case_observations_to_co_to_observation
from .obsname import get_RecNameList_and_FldTknList

from datasets.utils.logging import disable_progress_bar
# disable_progress_bar()

logging.basicConfig(level=logging.INFO, format='[%(levelname)s:%(asctime)s:(%(filename)s@%(lineno)d %(name)s)]: %(message)s')
logger = logging.getLogger(__name__)

class Tokenizer_Transform:
    def __init__(self, tokenizer_fn, idx2tkn, fldtkn_args):
        self.tokenizer_fn = tokenizer_fn
        self.idx2tkn = idx2tkn
        self.tkn2idx = {v:k for k, v in enumerate(idx2tkn)}
        self.fldtkn_args = fldtkn_args

    def __call__(self, examples):
        # TODO: what if it is for llm tools. 
        # maybe just one example each time?
        
        tokenizer_fn = self.tokenizer_fn
        df_rec = pd.DataFrame({k:v for k, v in examples.items()})
        df_output = pd.DataFrame([tokenizer_fn(rec, self.fldtkn_args) for idx, rec in df_rec.iterrows()])
        
        # these are the two new columns 
        df_output['tknidx'] = df_output['tkn'].apply(lambda x: [self.tkn2idx[i] for i in x])   
        df_output = df_output.drop(columns = ['tkn'])

        for k, v in df_output.to_dict('list').items():
            examples[self.fldtkn_args['Name'] + '_' + k] = v
        return examples


class DS_Rec_Info_Generator:
    def __init__(self, df_rec, RootID, RecDT):
        self.df_rec = df_rec
        self.RootID = RootID
        self.RecDT = RecDT

    def __call__(self):
        idx = 0
        for PID, df_rec_p in self.df_rec.groupby('PID'):
            d = {}
            d[f'{self.RootID}_idx'] = idx  
            d[self.RootID] = PID
            d['interval'] = min(df_rec_p.index), max(df_rec_p.index)
            if self.RecDT is not None:
                d['dates'] = [i.isoformat() for i in df_rec_p[self.RecDT].tolist()]
            idx = idx + 1
            yield d


def add_fldtkn_to_ds_rec(ds_rec, fldtkn_args):
    # small phi
    pypath = fldtkn_args['pypath']
    # logger.info(f'load fldtkn pipeline from: {pypath} ...')
    module = load_module_variables(pypath)
    tokenizer_fn = module.MetaDict['tokenizer_fn']

    idx2tkn = module.MetaDict['idx2tkn']
    # possible other information
    if 'column_to_top_values' in module.MetaDict:
        fldtkn_args['column_to_top_values'] = module.MetaDict['column_to_top_values']
    if 'item_to_configs' in module.MetaDict:
        fldtkn_args['item_to_configs'] = module.MetaDict['item_to_configs']

    # print(f'\n=================== process fldtkn: {FldTknName} ===================')
    tokenizer_transform = Tokenizer_Transform(tokenizer_fn, idx2tkn, fldtkn_args)
    
    batch_size = fldtkn_args.get('batch_size', 1000)
    ds_rec = ds_rec.map(tokenizer_transform, batched = True, num_proc=1, batch_size=batch_size)
    # logger.info(f'idx2tkn list: {idx2tkn} ...')
    # logger.info(f'ds_rec.column_names: {ds_rec.column_names} ...')
    return ds_rec 


def get_RecObsInfo_for_a_RecObsName(RecObsName,
                                    name_CasePhi,
                                    get_selected_columns, 
                                    cohort_args, 
                                    ckpd_to_CkpdObsConfig, 
                                    record_to_ds_rec = {}, 
                                    record_to_ds_rec_info = {}):
    
    RootID = cohort_args['RootID']
    # print(f'\n======================== RecObsName: {RecObsName} ========================')
    d = parse_RecObsName(RecObsName, ckpd_to_CkpdObsConfig)
    RecName = d['RecName']
    CkpdName = d['CkpdName']
    FldName = d['FldName']
    rec_args = load_record_args(RecName, cohort_args)

    run_fldtkn_on_the_fly = False 

    if FldName is not None:
        FldTknName = RecName + '-' + FldName + 'Tkn'
        fldtkn_args = load_fldtkn_args(RecName, FldTknName, cohort_args)
        # print(fldtkn_args['pypath'])
        module = load_module_variables(fldtkn_args['pypath'])
        item_to_configs = module.MetaDict.get('item_to_configs', {})
        column_to_top_values = module.MetaDict.get('column_to_top_values', {})
        fldtkn_args['Name'] = FldTknName
        fldtkn_args['item_to_configs'] = item_to_configs
        fldtkn_args['column_to_top_values'] = column_to_top_values
        fld_idx2tkn = module.idx2tkn 
        fld_tokenizer_fn = module.tokenizer_fn
        # print(FldTknName, ':', fld_idx2tkn[:10])
    else:
        FldTknName = None 
        fld_idx2tkn = None 
        fldtkn_args = None 
        fld_tokenizer_fn = None

    # get recname information
    if RecName in record_to_ds_rec:
        ds_rec = record_to_ds_rec[RecName]
        ds_rec_info = record_to_ds_rec_info[RecName]
    else:
        ################################################################################## to update
        try:
            ds_rec, ds_rec_info = load_ds_rec_and_info(RecName, cohort_args)
        except Exception as e:
            logger.info(f'Error in loading the dataset: {RecName} with error {e}')
            raise ValueError(f'Error in loading the dataset: {RecName}')
        ##################################################################################

    column_names = ds_rec.column_names
    selected_columns = get_selected_columns(RecObsName, column_names, cohort_args, rec_args, name_CasePhi)
    ds_rec = ds_rec.select_columns(selected_columns)
    df_rec_info = ds_rec_info.to_pandas().set_index(RootID)

    # what if the selected_columns doesn't exist in the dataset? on fldtkn calculate on the fly.
    if FldTknName is not None and FldTknName + '_tknidx' not in selected_columns:
        run_fldtkn_on_the_fly = True
        
    RecObsInfo = {
        'rec_args': rec_args,
        'RecName': RecName, 
        'CkpdName': CkpdName, 
        'CkpdInfo': ckpd_to_CkpdObsConfig[CkpdName] if CkpdName in ckpd_to_CkpdObsConfig else None,
        'FldName': FldName,
        'FldTknName': FldTknName,
        'FldIdx2Tkn': fld_idx2tkn,
        'fldtkn_args': fldtkn_args,
        'fld_tokenizer_fn': fld_tokenizer_fn,
        'run_fldtkn_on_the_fly': run_fldtkn_on_the_fly,
        'ds_rec': ds_rec,
        'df_rec_info': df_rec_info,
    }
    return RecObsInfo

    
def get_RecObsName_to_RecObsInfo(RecObsName_List, 
                                 name_CasePhi, 
                                 get_selected_columns, 
                                 cohort_args, 
                                 Ckpd_ObservationS, 
                                 record_to_ds_rec = {}, 
                                 record_to_ds_rec_info = {}):
    
    RecObsName_to_RecObsInfo = {}
    for RecObsName in RecObsName_List:
        RecObsInfo = get_RecObsInfo_for_a_RecObsName(RecObsName,
                                                     name_CasePhi, 
                                                     get_selected_columns, 
                                                     cohort_args, 
                                                     Ckpd_ObservationS, 
                                                     record_to_ds_rec, 
                                                     record_to_ds_rec_info)
        RecObsName_to_RecObsInfo[RecObsName] = RecObsInfo
    return RecObsName_to_RecObsInfo


def get_CaseObsInfo_for_a_CaseObsName(CaseObsName, 
                                      SPACE, 
                                      cohort_args, 
                                      record_to_ds_rec = {}, 
                                      record_to_ds_rec_info = {}):
    
    RecObsName_List, name_CasePhi = parse_CaseObsName(CaseObsName)
    Phi_Tools = fetch_caseobs_Phi_tools(name_CasePhi, CaseObsName, SPACE)

    get_selected_columns = Phi_Tools['get_selected_columns']
    ROName_to_ROInfo = get_RecObsName_to_RecObsInfo(RecObsName_List, 
                                                    name_CasePhi, 
                                                    get_selected_columns,
                                                    cohort_args, 
                                                    cohort_args['ckpd_to_CkpdObsConfig'], 
                                                    record_to_ds_rec, 
                                                    record_to_ds_rec_info)
    
    

    get_CO_vocab = Phi_Tools['get_CO_vocab']
    CO_vocab = get_CO_vocab(ROName_to_ROInfo)
    CaseObsInfo = {
        'RecObsName_List': RecObsName_List,
        'name_CasePhi': name_CasePhi,
        'get_selected_columns': get_selected_columns,
        'CaseObsName': CaseObsName,
        'ROName_to_ROInfo': ROName_to_ROInfo,
        'fn_CasePhi': Phi_Tools['fn_CasePhi'],
        'get_CO_id': Phi_Tools['get_CO_id'],
        'CO_Folder': Phi_Tools['CO_Folder'],
        'CO_vocab': CO_vocab,
    }
    return CaseObsInfo


def get_CaseObsName_to_CaseObsInfo(CaseObsName_List,
                                   SPACE, 
                                   cohort_args, 
                                   record_to_ds_rec = {}, 
                                   record_to_ds_rec_info = {}):
    CaseObsName_to_CaseObsInfo = {}
    for CaseObsName in CaseObsName_List:
        CaseObsInfo = get_CaseObsInfo_for_a_CaseObsName(CaseObsName, 
                                                       SPACE, 
                                                       cohort_args, 
                                                       record_to_ds_rec, 
                                                       record_to_ds_rec_info)
        CaseObsName_to_CaseObsInfo[CaseObsName] = CaseObsInfo
    return CaseObsName_to_CaseObsInfo


def get_CaseFeatInfo_for_a_CaseFeatName(name_CaseGamma, 
                                        case_observations,
                                        SPACE, 
                                        cohort_args, 
                                        record_to_ds_rec, 
                                        record_to_ds_rec_info):
    
    # COList_hash, name_CaseGamma = parse_CaseFeatName(CaseFeatName)
    co_to_COName, co_to_CONameInfo = convert_case_observations_to_co_to_observation(case_observations)
    COName_to_co = {v: k for k, v in co_to_COName.items()}
    PipelineInfo = get_RecNameList_and_FldTknList(co_to_CONameInfo, cohort_args['ckpd_to_CkpdObsConfig'])
    COName_List = [CaseName for co, CaseName in co_to_COName.items()]
    CaseFeatName = convert_CONameList_to_CFName(COName_List, name_CaseGamma)

    Gamma_Tools = fetch_casefeat_Gamma_tools(name_CaseGamma, CaseFeatName, SPACE)
    fn_CaseGamma = Gamma_Tools['fn_CaseGamma']
    get_CF_vocab = Gamma_Tools['get_CF_vocab']
    get_CF_id = Gamma_Tools['get_CF_id']
    COName_to_COInfo = get_CaseObsName_to_CaseObsInfo(COName_List,
                                                      SPACE, 
                                                      cohort_args, 
                                                      record_to_ds_rec, 
                                                      record_to_ds_rec_info)
    
    co_to_COvocab = {COName_to_co[COName]: CaseObsInfo['CO_vocab'] for COName, CaseObsInfo in COName_to_COInfo.items()}
    CF_vocab = get_CF_vocab(co_to_COvocab)
    CF_Folder = Gamma_Tools['CF_Folder']

    CaseFeatInfo = {
        'CaseFeatName': CaseFeatName,
        'name_CaseGamma': name_CaseGamma,
        'case_observations': case_observations, 
        'co_to_COName': co_to_COName,
        'COName_to_COInfo': COName_to_COInfo,
        'COName_List': COName_List,
        'PipelineInfo': PipelineInfo,
        'COName_List': COName_List,
        'COName_to_COInfo': COName_to_COInfo,
        'fn_CaseGamma': fn_CaseGamma,
        'get_CF_id': get_CF_id,
        'get_CF_vocab': get_CF_vocab,
        'CF_vocab': CF_vocab,
        'CF_Folder': CF_Folder,
    }
    return CaseFeatInfo
    


def get_fn_case_GammaFullInfo(Gamma_Config, 
                              base_config, 
                              RecName_to_dsRec, 
                              RecName_to_dsRecInfo, 
                              df_case_learning, 
                              use_CF_from_disk, 
                              use_CO_from_disk):

    CaseFeatInfo = get_CaseFeatInfo_for_a_CaseFeatName(Gamma_Config['name_CaseGamma'],
                                                       Gamma_Config['case_observations'],
                                                       base_config['SPACE'], 
                                                       base_config, 
                                                       RecName_to_dsRec, 
                                                       RecName_to_dsRecInfo)
    
    FnCaseFeatGamma = CaseFeatureTransformer(CaseFeatInfo['co_to_COName'],
                                             CaseFeatInfo['COName_to_COInfo'], 
                                             CaseFeatInfo['name_CaseGamma'], 
                                             CaseFeatInfo['fn_CaseGamma'], 
                                             CaseFeatInfo['CF_vocab'], 
                                             CaseFeatInfo['get_CF_id'],
                                             base_config,
                                             CaseFeatInfo['CF_Folder'], 
                                             df_case_learning,
                                             use_CF_from_disk, 
                                             use_CO_from_disk)
    
    CaseFeatInfo['FnCaseFeatGamma'] = FnCaseFeatGamma
    return CaseFeatInfo


class CaseObserverTransformer:
    def __init__(self, 
                 ro_to_ROName,
                 ROName_to_ROInfo, 
                 name_CasePhi, 
                 fn_CasePhi, 
                 CO_vocab, 
                 get_CO_id,
                 cohort_args,
                 CO_Folder = None, 
                 df_case = None,
                 use_CO_from_disk = False):
        
        self.ro_to_ROName = ro_to_ROName
        ROName_List = [i for i in ROName_to_ROInfo]
        self.COName = convert_RONameList_to_COName(ROName_List, name_CasePhi)

        self.ROName_to_ROInfo = ROName_to_ROInfo
        self.fn_CasePhi = fn_CasePhi
        self.CO_vocab = CO_vocab
        
        self.CO_Folder_data = os.path.join(CO_Folder, 'data')    
        self.CO_Folder_vocab = os.path.join(CO_Folder, 'vocab.p')
        self.get_CO_id = get_CO_id
        self.cohort_args = cohort_args

        # read the ds_CO_data and df_CO_info from the disk
        self.use_CO_from_disk = use_CO_from_disk
        if type(df_case) == pd.DataFrame:   
            self.COids = list(set(df_case.apply(lambda x: get_CO_id(x, self.COName, cohort_args), axis = 1).tolist()))
        else:
            self.COids = None 
        self.ds_CO_data, self.df_CO_info = self.load_COs_from_disk(self.CO_Folder_data, self.COids)
        
        # place holder for the new calculated COs
        self.new_COs = {}  
        self.MAX_NEW_CASEOBS_CACHE_SIZE = 50000

    def get_idx_to_CO_from_disk(self, COName, idx_to_examples, get_CO_id, ds_CO_data, df_CO_info, cohort_args):

        idx_to_examples_todo = {}
        idx_to_CO = {}

        for idx, case_example in idx_to_examples.items():
            
            caseobs_id = get_CO_id(case_example, COName, cohort_args)
            
            if caseobs_id in self.new_COs:
                logger.debug('1st: use cached new_calculated_caseobs')
                CaseObservation = self.new_COs[caseobs_id]
                idx_to_CO[idx] = CaseObservation
            
            elif caseobs_id in df_CO_info.index:
                logger.debug('1st: load from old caseobs')
                caseobs_info = df_CO_info.loc[caseobs_id]
                if len(caseobs_info) > 1:
                    caseobs_info = caseobs_info.iloc[0]
                    logger.warning(f'Warning: there are multiple caseobs_info for {COName}: {caseobs_id}, select the first one.')
                caseobs_idx_in_data = int(caseobs_info['caseobs_idx_in_data'])
                CaseObservation = ds_CO_data[caseobs_idx_in_data]
                caseobs_id_from_disk = CaseObservation.pop('caseobs_id')
                
                if caseobs_id != caseobs_id_from_disk:
                    logger.warning('Erorr: caseobs_id != CaseObservation.pop(caseobs_id)')
                    logger.warning(f'{caseobs_id} <--- caseobs_id')
                    logger.warning(CaseObservation)
                    logger.warning(ds_CO_data)
                    logger.warning(f'{caseobs_id_from_disk} <--- CaseObservation.pop(caseobs_id)')
                assert caseobs_id == caseobs_id_from_disk
                
                idx_to_CO[idx] = CaseObservation
            
            else:
                idx_to_examples_todo[idx] = case_example
        return idx_to_examples_todo, idx_to_CO

    @staticmethod
    def get_Record_P(idx_to_case_example, RecObsName_to_RecObsInfo):
        PIDValueS = [example['PID'] for idx, example in idx_to_case_example.items()]
        PIDValueS_Unique = list(set(PIDValueS))

        # print(len(PIDValueS_Unique))
        RecName_to_REC_P = {}

        for RecObsName, RecObsInfo in RecObsName_to_RecObsInfo.items():
            ######################
            # RecName = RecCkpd.split('_')[0]
            # RecName = RecObsName.split('-')[0]
            RecName = RecObsInfo['RecName']
            ######################

            # print(RecName)
            if RecName not in RecName_to_REC_P: 
                RecName_to_REC_P[RecName] = {} 
            
            ds_rec = RecObsInfo['ds_rec']
            df_rec_info = RecObsInfo['df_rec_info']
            for PIDValue in PIDValueS_Unique:
                if PIDValue in RecName_to_REC_P[RecName]: continue 

                ############## option 1
                # PIDidx = PID_to_idx[PIDValue]
                # example_p_info = ds_rec_info[PIDidx]

                ############## option 2
                if PIDValue not in df_rec_info.index: 
                    RecName_to_REC_P[RecName][PIDValue] = None 
                    continue

                info_p = df_rec_info.loc[PIDValue]

                pid_interval = info_p['interval']
                ds_rec_p = ds_rec.select(range(pid_interval[0], pid_interval[1] + 1))
                RecName_to_REC_P[RecName][PIDValue] = ds_rec_p

            # idx_to_DS_Rec_P[RecName] = [RecName_to_DS_REC_P[RecName][PIDValue] in enumerate(idx_to_PIDValueS)]
        return RecName_to_REC_P # , idx_to_DS_Rec_P  

    @staticmethod
    def get_idx_to_RecObsName_to_RecObsDS(idx_to_case_example, RecObsName_to_RecObsInfo, RecName_to_REC_P):

        # 1. PID, ObsDT
        # idx_to_PIDValue = case_examples['PID']      # p_i
        idx_to_PIDValue = {idx: example['PID'] for idx, example in idx_to_case_example.items()}

        
        # idx_to_ObsDTValue = case_examples['ObsDT'] # t_{ij}
        idx_to_ObsDTValue = {idx: example['ObsDT'] for idx, example in idx_to_case_example.items()}

        # idx_to_RecObsName_to_RecObsDS = []               # R_{ij}^{name, ckpd, fld}
        idx_to_RO_to_ROds = {}   

        # for idx, PIDValue in enumerate(idx_to_PIDValue):
        for idx, PIDValue in idx_to_PIDValue.items():
            # focus on one case (example)
            PIDValue   = idx_to_PIDValue[idx]   # p_i
            ObsDTValue = idx_to_ObsDTValue[idx] # t_{ij}

            RO_to_ROds = {} # <--- for one example (pid, pred_dt)
            for RecObsName, RecObsInfo in RecObsName_to_RecObsInfo.items():
                # ROinfo = RecObsInfo
                RecName = RecObsInfo['RecName']
                
                df_rec_info = RecObsInfo['df_rec_info'] # R_i
                ds_rec_p = RecName_to_REC_P[RecName][PIDValue] # R_i

                if ds_rec_p is None:
                    RO_to_ROds[RecObsName] = None
                    continue
                
                if len(ds_rec_p) == 0:
                    RO_to_ROds[RecObsName] = None
                    continue

                if RecObsInfo['CkpdInfo'] is None:
                    RO_to_ROds[RecObsName] = ds_rec_p
                    continue
                

                ########### safe checking ###########
                # if 'CkpdInfo' in RecObsInfo:
                # if 'dates'
                CkpdInfo = RecObsInfo['CkpdInfo']
                info_p = df_rec_info.loc[PIDValue]
                dates = info_p['dates'] # <--- will this still take time?

                assert len(dates) == len(ds_rec_p)
                # print('\n' + RecObsName)
                # print(idx_to_case_example[idx])
                
                # df_rec_p = ds_rec_p.to_pandas()
                # print(ds_rec_p)
                # print(df_rec_p.head())
                # print(df_rec_p['DT_s'].min(), '-----', df_rec_p['DT_s'].max())   
                # print(df_rec_p['DT_s'].iloc[0], '-----', df_rec_p['DT_s'].iloc[-1]) 
                # dates = info_p['dates']
                # print(dates[0], '----', dates[-1])

                # print(dates[0], '----', ds_rec_p[RecDT][0].isoformat())
                # print(dates[-1], '----', ds_rec_p[RecDT][-1].isoformat())

                # we first select row, and then select columns to get faster speed.
                # otherwise, 10x lower speed.
                RecDT = RecObsInfo['rec_args']['RecDT']
                assert dates[0] == ds_rec_p[0][RecDT].isoformat()
                assert dates[-1] == ds_rec_p[-1][RecDT].isoformat()
                

                ######### TO COMMENT OUT
                # print(pd.Series(ds_rec_p['PID']).value_counts())
                # print('ds_rec_p:')
                # print([i.isoformat() for i in ds_rec_p['DT_s'][:10]])
                # print([i.isoformat() for i in ds_rec_p['DT_s'][-10:]])
                # # print(ds_rec_p['DT_s'][-10:])
                # print('dates:')
                # print(dates[:10])
                # print(dates[-10:])
                #########
                
                # get the DT_s and DT_e 
                DistStartToPredDT = CkpdInfo['DistStartToPredDT']
                DistEndToPredDT = CkpdInfo['DistEndToPredDT']
                TimeUnit = CkpdInfo['TimeUnit']

                # raise the caution here: the datetime format need to be consistent. 
                DT_s = (ObsDTValue + pd.to_timedelta(DistStartToPredDT, unit = TimeUnit)).isoformat()
                DT_e = (ObsDTValue + pd.to_timedelta(DistEndToPredDT,   unit = TimeUnit)).isoformat()
                
                # get the idx_s and idx_e
                idx_s = find_timelist_index(dates, DT_s)
                idx_e = find_timelist_index(dates, DT_e)
                

                ############ TO COMMENT OUT
                # print(dates[0], dates[-1], idx_s, idx_e)
                # selected_dates = dates[idx_s: idx_e]
                # print(selected_dates[0], selected_dates[-1])
                ############
                
                # assert idx_s != idx_e
                    
                # only for certain RecName, need to pay attention for the future new record type. 
                ######################################################
                if idx_s == idx_e:
                    # this means that between idx_s and idx_e, there are no records.
                    # print('No Information')
                    RO_to_ROds[RecObsName] = None
                    continue
                    # raise ValueError(f'No Information, save idx_s and idx_e: {idx_s}')
                ######################################################
                # print('length of ds_rec_p:', ds_rec_p)
                ROds = ds_rec_p.select(range(idx_s, idx_e)) # R_{ij}^{ckpd, name, fld}
                # print(ROds)
                
                ############ This could take a lot of time #############
                # DT_s_selected, DT_e_selected = min(ROds[RecDT]), max(ROds[RecDT])
                DT_s_selected, DT_e_selected = ROds[0][RecDT], ROds[-1][RecDT]
                ######################################################  
                DT_s_selected, DT_e_selected = DT_s_selected.isoformat(), DT_e_selected.isoformat()
                
                if not (DT_s <= DT_s_selected and  DT_e_selected <= DT_e):
                    logger.warning(f'targeted range: {DT_s} - {DT_e}')
                    logger.warning(f'selected range: {DT_s_selected} - {DT_e_selected}')
                    raise ValueError('Error: the selected range is not in the targeted range.')
                
                
                if RecObsInfo['run_fldtkn_on_the_fly'] == True:
                    # fldtkn_args = RecObsInfo['fldtkn_args']
                    # fld_tokenizer_fn = RecObsInfo['fld_tokenizer_fn']
                    # ROds = ROds.map(lambda x: fld_tokenizer_fn(x, fldtkn_args))
                    fldtkn_args = RecObsInfo['fldtkn_args']
                    ROds = add_fldtkn_to_ds_rec(ROds, fldtkn_args)
                    
                RO_to_ROds[RecObsName] = ROds 
                
            # idx_to_RecObsName_to_RecObsDS.append(RecObsName_to_RecObsDS)
            idx_to_RO_to_ROds[idx] = RO_to_ROds 
        return idx_to_RO_to_ROds

    def __call__(self, case_examples):
        if len(self.new_COs) > self.MAX_NEW_CASEOBS_CACHE_SIZE:
            self.save_new_COs_to_disk(self.CO_Folder_data)
            self.ds_CO_data, self.df_CO_info = self.load_COs_from_disk(self.CO_Folder_data, self.COids)
            
        ROName_to_ROInfo = self.ROName_to_ROInfo
        COName = self.COName
        CO_vocab = self.CO_vocab
        get_CO_id = self.get_CO_id
        ds_CO_data = self.ds_CO_data
        df_CO_info = self.df_CO_info
        cohort_args = self.cohort_args
        
        # case_examples
        num_examples = len(case_examples[[i for i in case_examples.keys()][0]])
        idx_to_examples = {i: {k: v[i] for k, v in case_examples.items()} for i in range(num_examples)}

        idx_to_examples_todo, idx_to_CO = self.get_idx_to_CO_from_disk(COName, 
                                                                       idx_to_examples, 
                                                                       get_CO_id, 
                                                                       ds_CO_data, 
                                                                       df_CO_info, 
                                                                       cohort_args)
        
            
        # one should consider how to save the caseobs_id here to really same time. 
        RecName_to_REC_P = self.get_Record_P(idx_to_examples_todo, ROName_to_ROInfo)
        idx_to_ROName_to_ROds = self.get_idx_to_RecObsName_to_RecObsDS(idx_to_examples_todo, ROName_to_ROInfo, RecName_to_REC_P)

        for idx in idx_to_examples_todo:
            case_example = idx_to_examples_todo[idx]
            caseobs_id = get_CO_id(case_example, COName, cohort_args)
            ROName_to_ROds = idx_to_ROName_to_ROds[idx]

            if caseobs_id in self.new_COs:
                logger.debug('2nd: use cached new_calculated_caseobs')
                CaseObservation = self.new_COs[caseobs_id]
            else:
                logger.debug('2nd: calculate new caseobs')
                ##########################
                # a_{ij} = \Phi([R_{ij}^{name, ckpd, \phi_fld}])
                CaseObservation = self.fn_CasePhi(case_example, 
                                                  ROName_to_ROds, 
                                                  ROName_to_ROInfo, 
                                                  CO_vocab, 
                                                  cohort_args) 
                ##########################
                self.new_COs[caseobs_id] = CaseObservation

            idx_to_CO[idx] = CaseObservation

        # sort the idx_to_CaseObservation to the normal order
        idx_to_CO = dict(sorted(idx_to_CO.items()))
        df = pd.DataFrame([v for idx, v in idx_to_CO.items()])
        output = df.to_dict(orient = 'list')
        return output
    
    def load_COs_from_disk(self, CO_Folder_data, COids, coerce = False):
        ds_CO_data_empty = None 
        columns = ['caseobs_id', 'caseobs_idx_in_data']
        df_CO_info_empty = pd.DataFrame(columns = columns).set_index('caseobs_id')
          
        # case 1: we don't need to use the CO data from the disk
        if self.use_CO_from_disk is False and coerce == False: return ds_CO_data_empty, df_CO_info_empty

        if COids is not None: logger.info(f'provided caseobs_ids num: {len(COids)}')
        
        if not os.path.exists(CO_Folder_data): os.makedirs(CO_Folder_data)    
        
        set_list = os.listdir(CO_Folder_data)

        # case 1: no caseobs data save in CaseObsFolder_data
        if len(set_list) == 0: return ds_CO_data_empty, df_CO_info_empty

        # case 2: caseobs data save in CaseObsFolder_data
        Path_to_DS = {}
        for i in set_list:
            ds_data_path = os.path.join(CO_Folder_data, i)
            
            # read ds_data_path
            try:
                ds = datasets.Dataset.load_from_disk(ds_data_path)
            except:
                logger.warning(f'\nError in the folder, skip it and remove it: {ds_data_path}')
                os.remove(ds_data_path); continue 
            
            # fetch caseobs_id_from_disk from ds
            caseobs_id_from_disk = set(ds.select_columns(['caseobs_id']).to_pandas()['caseobs_id'].unique())
                
            # check whether there are overlap between caseobs_ids and caseobs_id_from_disk, if these are no overlap, skip it. 
            if COids is not None:
                overlap = caseobs_id_from_disk.intersection(COids)
                if len(overlap) == 0: continue

            # append it to ds_data_path.
            logger.info(f'get pre-calcuated CO from: {ds_data_path}')
            Path_to_DS[ds_data_path] = ds
            
        # case 2.1: still no information
        if len(Path_to_DS) == 0: return ds_CO_data_empty, df_CO_info_empty
    
        # case 2.2: concatenate the datasets and save to the disk
        ds_CO_data = datasets.concatenate_datasets([v for _, v in Path_to_DS.items()])
        df_CO_info = ds_CO_data.select_columns(['caseobs_id']).to_pandas().reset_index()
        df_CO_info = df_CO_info.set_index('caseobs_id').rename(columns = {'index': 'caseobs_idx_in_data'})
        return ds_CO_data, df_CO_info

    def save_new_COs_to_disk(self, CO_Folder_data = None):
        if CO_Folder_data is None: CO_Folder_data = self.CO_Folder_data
        
        if len(self.new_COs) == 0:
            # logger.info('No new calculated caseobs'); 
            self.new_COs = {}; return None 
        
        if self.use_CO_from_disk == False:
            # logger.info('use_CO_from_disk is false and empty new_COs'); 
            self.new_COs = {}; return None

        # get the new data and information
        df_CO_new = pd.DataFrame([add_key_return_dict(v, 'caseobs_id', k) for k, v  in self.new_COs.items()])
        ds_CO_new = datasets.Dataset.from_pandas(df_CO_new)

        # we only save the new caseobs
        time.sleep(np.random.rand())
        time_finger = datetime.now().isoformat()[:-4].replace(':', '-') + str(round(np.random.rand()* 100, 0))
        
        # save new COs to the disk
        new_path = os.path.join(CO_Folder_data, f'set-dt{time_finger}-sz{len(ds_CO_new)}')
        ds_CO_new.save_to_disk(new_path)
        self.new_COs = {}; return None
        
    def clean_and_update_COs_in_disk(self, CO_Folder_data):
        if not os.path.exists(CO_Folder_data): return None   
        
        # get the set_list
        set_list = os.listdir(CO_Folder_data)
        if len(set_list) == 0: return None

        Path_to_SmallDS = {}
        for i in set_list:
            # ds set path
            size = i.split('-')[-1]
            assert 'sz' in size 
            size = int(size.split('sz')[-1])
            if size >= 100000: continue
            ds_data_path = os.path.join(CO_Folder_data, i)
            
            try:
                ds = datasets.Dataset.load_from_disk(ds_data_path)
            except:
                logger.warning(f'\nError in the folder, remove it and skip it: {ds_data_path}')
                os.remove(ds_data_path); continue 
            
            Path_to_SmallDS[ds_data_path] = ds

        # ------------- TO UPDATE: the new version in case df_caseobs_not that big -------------
        if len(Path_to_SmallDS) == 1: return None
        ds_COs = concatenate_datasets([v for _, v in Path_to_SmallDS.items()])
        time.sleep(np.random.rand())
        time_finger = datetime.now().isoformat()[:-4].replace(':', '-') + str(round(np.random.rand()* 100, 0))
        new_path = os.path.join(self.CaseObsFolder_data, f'set-dt{time_finger}-sz{len(ds_COs)}')
        logger.info(f'save a group of small ds as a big ds to new_path: {new_path}')
        ds_COs.save_to_disk(new_path)

        # remove the old_path
        for old_path, old_ds in Path_to_SmallDS.items():
            if len(old_ds) >= 1000000: continue 
            del old_ds
            if os.path.exists(old_path):  # Check if the path exists
                try:
                    shutil.rmtree(old_path)   # This removes the directory and all its contents
                except:
                    logger.warning(f"Fail to remove the path: {old_path}")
                    time.sleep(3)
                    shutil.rmtree(old_path)
            else:
                logger.warning(f"The path {old_path} does not exist. might be deleted by other process.")
        
        
class CaseFeatureTransformer:
    def __init__(self, 
                 co_to_COName,
                 COName_to_COInfo, 
                 name_CaseGamma, 
                 fn_CaseGamma, 
                 CF_vocab, 
                 get_CF_id,
                 cohort_args,
                 CF_Folder = None,
                 df_case = None,
                 use_CF_from_disk = False, 
                 use_CO_from_disk = False):

        # we have short name co for CaseObsName. 
        self.co_to_COName = co_to_COName
        self.COName_to_co = {v: k for k, v in self.co_to_COName.items()}

        # prepare the information for the CaseFeatureTransformer
        COName_list = [COName for COName in COName_to_COInfo]
        self.COName_to_COInfo = COName_to_COInfo
        self.CFName = convert_CONameList_to_CFName(COName_list, name_CaseGamma)
        self.fn_CaseGamma = fn_CaseGamma
        self.CF_vocab = CF_vocab
        self.co_to_COvocab = {self.COName_to_co[COName]: COInfo['CO_vocab'] for COName, COInfo in COName_to_COInfo.items()}
        
        self.CF_Folder_data  = os.path.join(CF_Folder, 'data')    
        self.CF_Folder_vocab = os.path.join(CF_Folder, 'vocab.p')
        self.get_CF_id = get_CF_id
        self.cohort_args = cohort_args

        # read the ds_CO_data and df_CO_info from the disk
        self.use_CF_from_disk = use_CF_from_disk

        if type(df_case) == pd.DataFrame:   
            self.CFids = list(set(df_case.apply(lambda x: get_CF_id(x, self.CFName, cohort_args), axis = 1).tolist()))
        else:
            self.CFids = None 

        self.ds_CF_data, self.df_CF_info = self.load_CFs_from_disk(self.CF_Folder_data, self.CFids)
        
        # place holder for the new calculated COs
        self.new_CFs = {}  
        self.MAX_NEW_CASEFEAT_CACHE_SIZE = 50000

        # the attributes to hold the fn_caseobs_phi. 
        self.COName_to_FnCaseObsPhi = {}
        for COName, COInfo in COName_to_COInfo.items():
            ro_to_ROName = None 
            FnCaseObsPhi = CaseObserverTransformer(ro_to_ROName,
                                                    COInfo['ROName_to_ROInfo'], 
                                                    COInfo['name_CasePhi'], 
                                                    COInfo['fn_CasePhi'], 
                                                    COInfo['CO_vocab'], 
                                                    COInfo['get_CO_id'],
                                                    cohort_args,
                                                    COInfo['CO_Folder'], 
                                                    df_case,
                                                    use_CO_from_disk)
            self.COName_to_FnCaseObsPhi[COName] = FnCaseObsPhi

    def get_idx_to_CF_from_disk(self, CFName, idx_to_examples, get_CF_id, ds_CF_data, df_CF_info, cohort_args):

        idx_to_examples_todo = {}
        idx_to_CF = {}

        for idx, case_example in idx_to_examples.items():
            casefeat_id = get_CF_id(case_example, CFName, cohort_args)
            
            if casefeat_id in self.new_CFs:
                logger.debug('1st: use cached new_CFs')
                CaseFeature = self.new_CFs[casefeat_id]
                idx_to_CF[idx] = CaseFeature
            
            elif casefeat_id in df_CF_info.index:
                logger.debug('1st: load from old case feats')
                casefeat_info = df_CF_info.loc[casefeat_id]
                if len(casefeat_info) > 1:
                    casefeat_info = casefeat_info.iloc[0]
                    logger.warning(f'Warning: there are multiple casefeat_info for {CFName}: {casefeat_id}, select the first one.')
                casefeat_idx_in_data = int(casefeat_info['casefeat_idx_in_data'])
                CaseFeature = ds_CF_data[casefeat_idx_in_data]
                casefeat_id_from_disk = CaseFeature.pop('casefeat_id')
                
                if casefeat_id != casefeat_id_from_disk:
                    logger.warning('Erorr: casefeat_id != CaseFeature.pop(casefeat_id)')
                    logger.warning(f'{casefeat_id} <--- casefeat_id')
                    logger.warning(CaseFeature)
                    logger.warning(ds_CF_data)
                    logger.warning(f'{casefeat_id_from_disk} <--- CaseFeature.pop(casefeat_id)')
                assert casefeat_id == casefeat_id_from_disk
                
                idx_to_CF[idx] = CaseFeature
            
            else:
                idx_to_examples_todo[idx] = case_example
        return idx_to_examples_todo, idx_to_CF
    
    @staticmethod
    def fetch_examples_with_complete_COs(idx_to_examples, 
                                         COName_to_co, 
                                         COName_to_COInfo, 
                                         COName_to_FnCaseObsPhi):
        
        ######## check the readiness of COs in these examples ########
        for COName, COInfo in COName_to_COInfo.items():
            co = COName_to_co[COName]
            CO_vocab = COInfo['CO_vocab']
            
            # CO_vocab must can the SeqTypes in case_Phi's output.
            example = idx_to_examples[[i for i in idx_to_examples][0]]
            if all([co+'_'+k in example for k in CO_vocab]): continue 


            FnCaseObsPhi = COName_to_FnCaseObsPhi[COName]

            # original idx list
            idx_list = [i for i in idx_to_examples]

            # get new caseobs to examples and add them to examples.
            # here examples is a dictionary.
            examples = pd.DataFrame([v for idx, v in idx_to_examples.items()]).to_dict(orient = 'list')
            examples['ObsDT'] = pd.to_datetime(examples['ObsDT'])
            # get caseobs for the examples
            examples_new = FnCaseObsPhi(examples)
            # update caseobs name with co
            examples_new = {co+'_'+k: v for k, v in examples_new.items()}
            # add caseobs to examples
            for k, v in examples_new.items(): examples[k] = v

            examples_list = [{k: v[i] for k, v in examples.items()} for i in range(len(idx_list))]
            idx_to_examples = {idx_list[i]: examples_list[i] for i in range(len(idx_list))}
        return idx_to_examples, COName_to_FnCaseObsPhi

    def __call__(self, case_examples):

        if len(self.new_CFs) > self.MAX_NEW_CASEFEAT_CACHE_SIZE:
            self.save_new_CFs_to_disk(self.CF_Folder_data)
            self.ds_CF_data, self.df_CF_info = self.load_CFs_from_disk(self.CF_Folder_data, self.CFids)
            
    
        COName_to_COInfo = self.COName_to_COInfo
        CFName = self.CFName
        CF_vocab = self.CF_vocab
        get_CF_id = self.get_CF_id
        ds_CF_data = self.ds_CF_data
        df_CF_info = self.df_CF_info
        cohort_args = self.cohort_args
        
        # case_examples
        num_examples = len(case_examples[[i for i in case_examples.keys()][0]])
        idx_to_examples = {i: {k: v[i] for k, v in case_examples.items()} for i in range(num_examples)}

        idx_to_examples_todo, idx_to_CF = self.get_idx_to_CF_from_disk(CFName, 
                                                                       idx_to_examples, 
                                                                       get_CF_id, 
                                                                       ds_CF_data, 
                                                                       df_CF_info, 
                                                                       cohort_args)
        
        # case_examples check whether it is in new_COs
        for idx in idx_to_examples_todo:
            case_example = idx_to_examples_todo[idx]
            casefeat_id = get_CF_id(case_example, CFName, cohort_args)
            if casefeat_id not in self.new_CFs: continue 
            logger.debug('2nd: use cached new_calculated_casefeats')
            CaseFeature = self.new_CFs[casefeat_id]
            idx_to_CF[idx] = CaseFeature
            idx_to_examples_todo.pop(idx)
        
        if len(idx_to_examples_todo) > 0: 
            number_todo = len( idx_to_examples_todo)
            COName_to_co = self.COName_to_co
            use_CO_from_disk = self.use_CF_from_disk
            results = self.fetch_examples_with_complete_COs(idx_to_examples_todo, 
                                                            COName_to_co, 
                                                            COName_to_COInfo, 
                                                            self.COName_to_FnCaseObsPhi)
            idx_to_examples_todo, self.COName_to_FnCaseObsPhi = results
            assert len(idx_to_examples_todo) == number_todo

        # idx_to_examples_todo
        co_to_COvocab = self.co_to_COvocab
        for idx in idx_to_examples_todo:
            logger.debug('3rd: calculate new caseobs')
            case_example = idx_to_examples_todo[idx]
            casefeat_id = get_CF_id(case_example, CFName, cohort_args)
            ##########################
            CaseFeature = self.fn_CaseGamma(case_example, co_to_COvocab, CF_vocab, cohort_args) 
            ##########################
            self.new_CFs[casefeat_id] = CaseFeature
            idx_to_CF[idx] = CaseFeature

        # sort the idx_to_CaseObservation to the normal order
        idx_to_CF = dict(sorted(idx_to_CF.items()))
        df = pd.DataFrame([v for idx, v in idx_to_CF.items()])
        output = df.to_dict(orient = 'list')
        return output

    def load_CFs_from_disk(self, CF_Folder_data, CFids = None):
        ds_CF_data_empty = None 
        columns = ['casefeat_id', 'casefeat_idx_in_data']
        df_CF_info_empty = pd.DataFrame(columns = columns).set_index('casefeat_id')
          
        # case 1: we don't need to use the CO data from the disk
        if self.use_CF_from_disk is False: return ds_CF_data_empty, df_CF_info_empty

        if CFids is not None: logger.info(f'provided casefeat_ids num: {len(CFids)}')
        
        if not os.path.exists(CF_Folder_data): os.makedirs(CF_Folder_data)    
        
        set_list = os.listdir(CF_Folder_data)

        # case 1: no caseobs data save in CaseObsFolder_data
        if len(set_list) == 0: return ds_CF_data_empty, df_CF_info_empty

        # case 2: caseobs data save in CaseObsFolder_data
        Path_to_DS = {}
        for i in set_list:
            ds_data_path = os.path.join(CF_Folder_data, i)
            
            # read ds_data_path
            try:
                ds = datasets.Dataset.load_from_disk(ds_data_path)
            except:
                logger.warning(f'Error in the folder, skip it and remove it: {ds_data_path}')
                os.remove(ds_data_path); continue 
            
            # fetch caseobs_id_from_disk from ds
            caseobs_id_from_disk = set(ds.select_columns(['casefeat_id']).to_pandas()['casefeat_id'].unique())
                
            # check whether there are overlap between caseobs_ids and caseobs_id_from_disk, if these are no overlap, skip it. 
            if CFids is not None:
                overlap = caseobs_id_from_disk.intersection(CFids)
                if len(overlap) == 0: continue

            # append it to ds_data_path.
            logger.info(f'get pre-calcuated CF from: {ds_data_path}')
            Path_to_DS[ds_data_path] = ds
            
        # case 2.1: still no information
        if len(Path_to_DS) == 0: return ds_CF_data_empty, df_CF_info_empty
    
        # case 2.2: concatenate the datasets and save to the disk
        logger.info(f'the final size of the pre-calcuated CFs: {len(Path_to_DS)}')
        ds_CF_data = datasets.concatenate_datasets([v for _, v in Path_to_DS.items()])
        df_CF_info = ds_CF_data.select_columns(['casefeat_id']).to_pandas().reset_index()
        df_CF_info = df_CF_info.set_index('casefeat_id').rename(columns = {'index': 'casefeat_idx_in_data'})
        return ds_CF_data, df_CF_info
    
    def save_new_CFs_to_disk(self, CF_Folder_data = None):
        if CF_Folder_data is None: CF_Folder_data = self.CF_Folder_data

        if len(self.new_CFs) == 0:
            # logger.info('No new calculated casefeat'); 
            self.new_CFs = {}; return None 

        if self.use_CF_from_disk == False:
            # logger.info('use_CF_from_disk is false and empty new_CFs'); 
            self.new_CFs = {}; return None

        # get the new data and information
        df_CF_new = pd.DataFrame([add_key_return_dict(v, 'casefeat_id', k) for k, v  in self.new_CFs.items()])
        ds_CF_new = datasets.Dataset.from_pandas(df_CF_new)

        # we only save the new caseobs
        time.sleep(np.random.rand())
        time_finger = datetime.now().isoformat()[:-4].replace(':', '-') + str(round(np.random.rand()* 100, 0))
        
        # save new CFs to the disk
        new_path = os.path.join(CF_Folder_data, f'set-dt{time_finger}-sz{len(ds_CF_new)}')
        ds_CF_new.save_to_disk(new_path)
        self.new_CFs = {}; return None

    def clean_and_update_CFs_in_disk(self, CF_Folder_data):
        if not os.path.exists(CF_Folder_data): return None   
        
        # get the set_list
        set_list = os.listdir(CF_Folder_data)
        if len(set_list) == 0: return None

        Path_to_SmallDS = {}
        for i in set_list:
            # ds set path
            size = i.split('-')[-1]
            assert 'sz' in size 
            size = int(size.split('sz')[-1])
            if size >= 100000: continue
            ds_data_path = os.path.join(CF_Folder_data, i)
            
            try:
                ds = datasets.Dataset.load_from_disk(ds_data_path)
            except:
                logger.warning(f'\nError in the folder, remove it and skip it: {ds_data_path}')
                os.remove(ds_data_path); continue 
            
            Path_to_SmallDS[ds_data_path] = ds

        # ------------- TO UPDATE: the new version in case df_caseobs_not that big -------------
        if len(Path_to_SmallDS) == 1: return None
        ds_CFs = concatenate_datasets([v for _, v in Path_to_SmallDS.items()])
        time.sleep(np.random.rand())
        time_finger = datetime.now().isoformat()[:-4].replace(':', '-') + str(round(np.random.rand()* 100, 0))
        new_path = os.path.join(self.CaseObsFolder_data, f'set-dt{time_finger}-sz{len(ds_CFs)}')
        logger.info(f'save a group of small ds as a big ds to new_path: {new_path}')
        ds_CFs.save_to_disk(new_path)

        # remove the old_path
        for old_path, old_ds in Path_to_SmallDS.items():
            if len(old_ds) >= 1000000: continue 
            del old_ds
            if os.path.exists(old_path):  # Check if the path exists
                try:
                    shutil.rmtree(old_path)   # This removes the directory and all its contents
                except:
                    logger.warning(f"Fail to remove the path: {old_path}")
                    time.sleep(3)
                    shutil.rmtree(old_path)
            else:
                logger.warning(f"The path {old_path} does not exist. might be deleted by other process.")
        