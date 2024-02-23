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
from .loadtools import load_module_variables, add_key_return_dict, load_ds_rec_and_info
from .obsname import parse_RecObsName, convert_RecObsName_and_CaseTkn_to_CaseObsName


logging.basicConfig(level=logging.INFO, format='[%(levelname)s:%(asctime)s:(%(filename)s@%(lineno)d %(name)s)]: %(message)s')
logger = logging.getLogger(__name__)



def get_RecObsName_to_RecObsInfo(Record_Observations_List, 
                                 CaseTkn, 
                                 get_selected_columns, 
                                 cohort_args, 
                                 Ckpd_ObservationS, 
                                 record_to_ds_rec = {}, 
                                 record_to_ds_rec_info = {}):
    
    RootID = cohort_args['RootID']
    RecObsName_to_RecObsInfo = {}

    SPACE = cohort_args['SPACE']

    # for RO in RO_List:
    for RecObs_Name in Record_Observations_List:
        # print(f'\n======================== RecObs_Name: {RecObs_Name} ========================')
        d = parse_RecObsName(RecObs_Name)
        RecName = d['RecName']
        CkpdName = d['CkpdName']
        FldName = d['FldName']
        # print(RecName, CkpdName, FldName)
        rec_args = load_record_args(RecName, cohort_args)
        # RecDT = rec_args['RecDT']

        if FldName is not None:
            FldTknName = RecName + '-' + FldName + 'Tkn'
            fldtkn_args = load_fldtkn_args(RecName, FldTknName, cohort_args)
            # print(fldtkn_args['pypath'])
            module = load_module_variables(fldtkn_args['pypath'])
            fld_idx2tkn = module.idx2tkn 
            # print(FldTknName, ':', fld_idx2tkn[:10])
        else:
            FldTknName = None 
            fld_idx2tkn = None 
            
        # print(RecName, CkpdName, FldName, FldTknName)
        # print([i for i in record_to_ds_rec])
        if RecName in record_to_ds_rec:
            ds_rec = record_to_ds_rec[RecName]
            ds_rec_info = record_to_ds_rec_info[RecName]
            df_rec_info = ds_rec_info.to_pandas().set_index(RootID)
        else:
            ################################################################################## to update
            # hfds_folder = cohort_args['hfds_folder']
            # ds_path = os.path.join(hfds_folder, RecName)
            try:
                ds_rec, ds_rec_info = load_ds_rec_and_info(RecName, cohort_args)
            except Exception as e:
                logger.info(f'Error in loading the dataset: {RecName} with error {e}')
                raise ValueError(f'Error in loading the dataset: {RecName}')
            # ds_rec = datasets.Dataset.load_from_disk(ds_path)
            column_names = ds_rec.column_names
            selected_columns = get_selected_columns(RecObs_Name, column_names, cohort_args, rec_args, CaseTkn)
            ds_rec = ds_rec.select_columns(selected_columns)
            # ds_rec_info = datasets.Dataset.load_from_disk(ds_path + '_info')
            df_rec_info = ds_rec_info.to_pandas().set_index(RootID)
            # ds_rec, ds_rec_info = load_hfds_rec()
            ##################################################################################

        RecObsInfo = {
            'rec_args': rec_args,
            'RecName': RecName, 
            'CkpdName': CkpdName, 
            'CkpdInfo': Ckpd_ObservationS[CkpdName] if CkpdName in Ckpd_ObservationS else None,
            'FldName': FldName,
            'FldTknName': FldTknName,
            'FldIdx2Tkn': fld_idx2tkn,
            'ds_rec': ds_rec,
            'df_rec_info': df_rec_info,
        }
        RecObsName_to_RecObsInfo[RecObs_Name] = RecObsInfo

    return RecObsName_to_RecObsInfo


def find_timelist_index(dates: List[datetime], DT: datetime) -> int:
    low, high = 0, len(dates) - 1
    if DT < dates[0]:
        # return -1  # DT is smaller than the first date in the list
        return 0     # DT is smaller than the first date in the list
    if DT > dates[-1]:
        # return len(dates)  # DT is larger or equal to the last date in the list
        return len(dates)    # DT is larger or equal to the last date in the list

    while low <= high:
        mid = (low + high) // 2
        if dates[mid] < DT:
            low = mid + 1
        else:
            high = mid - 1
    return low


def get_Record_P(idx_to_case_example, RecObsName_to_RecObsInfo):
    PIDValueS = [example['PID'] for idx, example in idx_to_case_example.items()]
    PIDValueS_Unique = list(set(PIDValueS))

    # print(len(PIDValueS_Unique))
    RecName_to_REC_P = {}
    # P_to_REC_NameDict = {}
    # idx_to_DS_Rec_P = {}

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
 

def get_idx_to_RecObsName_to_RecObsDS(idx_to_case_example, RecObsName_to_RecObsInfo, RecName_to_REC_P):

    # 1. PID, ObsDT
    # idx_to_PIDValue = case_examples['PID']      # p_i
    idx_to_PIDValue = {idx: example['PID'] for idx, example in idx_to_case_example.items()}

    
    # idx_to_ObsDTValue = case_examples['ObsDT'] # t_{ij}
    idx_to_ObsDTValue = {idx: example['ObsDT'] for idx, example in idx_to_case_example.items()}

    # idx_to_RecObsName_to_RecObsDS = []               # R_{ij}^{name, ckpd, fld}
    idx_to_RecObsName_to_RecObsDS = {}   

    # for idx, PIDValue in enumerate(idx_to_PIDValue):
    for idx, PIDValue in idx_to_PIDValue.items():
        # focus on one case (example)
        PIDValue   = idx_to_PIDValue[idx]   # p_i
        ObsDTValue = idx_to_ObsDTValue[idx] # t_{ij}

        RecObsName_to_RecObsDS = {} # <--- for one example (pid, pred_dt)
        for RecObsName, RecObsInfo in RecObsName_to_RecObsInfo.items():
            RecName = RecObsInfo['RecName']
            df_rec_info = RecObsInfo['df_rec_info'] # R_i
            ds_rec_p = RecName_to_REC_P[RecName][PIDValue] # R_i

            if ds_rec_p is None:
                RecObsName_to_RecObsDS[RecObsName] = None
                continue
            
            if len(ds_rec_p) == 0:
                RecObsName_to_RecObsDS[RecObsName] = None
                continue

            if RecObsInfo['CkpdInfo'] is None:
                RecObsName_to_RecObsDS[RecObsName] = ds_rec_p
                continue
            
            CkpdInfo = RecObsInfo['CkpdInfo']
            info_p = df_rec_info.loc[PIDValue]
            dates = info_p['dates'] # <--- will this still take time?

            
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
            # assert idx_s != idx_e
                
            # only for certain RecName, need to pay attention for the future new record type. 
            ######################################################
            if idx_s == idx_e:
                # this means that between idx_s and idx_e, there are no records.
                # print('No Information')
                RecObsName_to_RecObsDS[RecObsName] = None
                continue
                # raise ValueError(f'No Information, save idx_s and idx_e: {idx_s}')
            ######################################################
            ds_p_ckpd_rec_fld = ds_rec_p.select(range(idx_s, idx_e)) # R_{ij}^{ckpd, name, fld}
            RecObsName_to_RecObsDS[RecObsName] = ds_p_ckpd_rec_fld 

        # idx_to_RecObsName_to_RecObsDS.append(RecObsName_to_RecObsDS)
        idx_to_RecObsName_to_RecObsDS[idx] = RecObsName_to_RecObsDS
    return idx_to_RecObsName_to_RecObsDS


class CaseObserverTransformer:
    def __init__(self, 
                 RecObsName_to_RecObsInfo, 
                 CaseTkn, get_casetkn_vocab, fn_CaseTkn, 
                 get_caseobs_id,
                 use_caseobs_from_disk = False,
                 CaseObsFolder = None, 
                 caseobs_ids = None):
        
        # self.test = test

        Record_Observations_List = [i for i in RecObsName_to_RecObsInfo]
        self.CaseObsName = convert_RecObsName_and_CaseTkn_to_CaseObsName(Record_Observations_List, CaseTkn)

        self.RecObsName_to_RecObsInfo = RecObsName_to_RecObsInfo
        self.fn_CaseTkn = fn_CaseTkn
        self.get_casetkn_vocab = get_casetkn_vocab
        self.CaseTknVocab = get_casetkn_vocab(RecObsName_to_RecObsInfo)
        
        self.CaseObsFolder_data = os.path.join(CaseObsFolder, 'data')    
        self.CaseObsFolder_vocab = os.path.join(CaseObsFolder, 'vocab.p')
        self.get_caseobs_id = get_caseobs_id

        # if the new caseobs are too many. 
        self.caseobs_ids = caseobs_ids
        self.use_caseobs_from_disk = use_caseobs_from_disk
        
        # self.clean_and_update_ds_caseobs_data_from_disk(self.CaseObsFolder_data)
        self.ds_caseobs_data, self.df_caseobs_info = self.load_ds_caseobs_data_from_disk(self.CaseObsFolder_data, self.caseobs_ids)
        self.new_calculated_caseobs = {}  
        self.MAX_NEW_CASEOBS_CACHE_SIZE = 50000


    def __call__(self, case_examples):
        if len(self.new_calculated_caseobs) > self.MAX_NEW_CASEOBS_CACHE_SIZE:
            self.save_new_caseobs_to_ds_caseobs()
            self.ds_caseobs_data, self.df_caseobs_info = self.load_ds_caseobs_data_from_disk(self.CaseObsFolder_data, self.caseobs_ids)
            self.new_calculated_caseobs = {}  

        CaseObsName = self.CaseObsName
        RecObsName_to_RecObsInfo = self.RecObsName_to_RecObsInfo
        CaseTknVocab = self.CaseTknVocab
        get_caseobs_id = self.get_caseobs_id
        ds_caseobs_data = self.ds_caseobs_data
        df_caseobs_info = self.df_caseobs_info
        
        # case_examples
        length = len(case_examples[list(case_examples.keys())[0]])
        case_examples_list = [{k: v[i] for k, v in case_examples.items()} for i in range(length)]
        idx_to_examples = {i: case_examples_list[i] for i in range(length)}
        
        # idx_to_examples_todo.
        idx_to_examples_todo = {}
        idx_to_CaseObservation = {}
        for idx, case_example in idx_to_examples.items():
            caseobs_id = get_caseobs_id(case_example, CaseObsName)
            if caseobs_id in self.new_calculated_caseobs:
                logger.debug('1st: use cached new_calculated_caseobs')
                CaseObservation = self.new_calculated_caseobs[caseobs_id]
                idx_to_CaseObservation[idx] = CaseObservation
            elif caseobs_id in df_caseobs_info.index:
                logger.debug('1st: load from old caseobs')
                caseobs_info = df_caseobs_info.loc[caseobs_id]
                caseobs_idx_in_data = int(caseobs_info['caseobs_idx_in_data'])
                CaseObservation = ds_caseobs_data[caseobs_idx_in_data]
                caseobs_id_from_disk = CaseObservation.pop('caseobs_id')
                if caseobs_id != caseobs_id_from_disk:
                    logger.warning('Erorr: caseobs_id != CaseObservation.pop(caseobs_id)')
                    logger.warning(f'{caseobs_id} <--- caseobs_id')
                    logger.warning(CaseObservation)
                    logger.warning(ds_caseobs_data)
                    logger.warning(f'{caseobs_id_from_disk} <--- CaseObservation.pop(caseobs_id)')
                assert caseobs_id == caseobs_id_from_disk
                idx_to_CaseObservation[idx] = CaseObservation
            else:
                idx_to_examples_todo[idx] = case_example
            
        # one should consider how to save the caseobs_id here to really same time. 
        RecName_to_REC_P = get_Record_P(idx_to_examples_todo, RecObsName_to_RecObsInfo)
        idx_to_RecObsName_to_RecObsDS = get_idx_to_RecObsName_to_RecObsDS(idx_to_examples_todo, RecObsName_to_RecObsInfo, RecName_to_REC_P)

        # idx_to_CaseObservation = []
        for idx, RecObsName_to_RecObsDS in idx_to_RecObsName_to_RecObsDS.items():
            # case_example = {k: v[idx] for k, v in case_examples.items()}
            case_example = idx_to_examples_todo[idx]
            caseobs_id = get_caseobs_id(case_example, CaseObsName)

            if caseobs_id in self.new_calculated_caseobs:
                logger.debug('2nd: use cached new_calculated_caseobs')
                CaseObservation = self.new_calculated_caseobs[caseobs_id]
            else:
                logger.debug('2nd: calculate new caseobs')
                ##########################
                # a_{ij} = \Phi([R_{ij}^{name, ckpd, \phi_fld}])
                CaseObservation = self.fn_CaseTkn(case_example, RecObsName_to_RecObsDS, RecObsName_to_RecObsInfo, CaseTknVocab) 
                ##########################
                self.new_calculated_caseobs[caseobs_id] = CaseObservation

            # idx_to_CaseObservation.append(CaseObservation)
            idx_to_CaseObservation[idx] = CaseObservation

        # sort the idx_to_CaseObservation to the normal order
        idx_to_CaseObservation = dict(sorted(idx_to_CaseObservation.items()))
        df = pd.DataFrame([v for idx, v in idx_to_CaseObservation.items()])
        output = df.to_dict(orient = 'list')
        return output
    
    def load_ds_caseobs_data_from_disk(self, CaseObsFolder_data, caseobs_ids):
        if self.use_caseobs_from_disk is False:
            ds_caseobs_data = None 
            df_caseobs_info = pd.DataFrame(columns = ['caseobs_id', 'caseobs_idx_in_data']).set_index('caseobs_id')
            return ds_caseobs_data, df_caseobs_info

        if caseobs_ids is not None: logger.info(f'caseobs_ids num: {len(caseobs_ids)}')
        if not os.path.exists(CaseObsFolder_data): os.makedirs(CaseObsFolder_data)    
        
        # get the set_list
        set_list = os.listdir(CaseObsFolder_data)

        # case 1: no caseobs data save in CaseObsFolder_data
        if len(set_list) == 0:
            # max_set = 0
            ds_caseobs_data = None 
            df_caseobs_info = pd.DataFrame(columns = ['caseobs_id', 'caseobs_idx_in_data']).set_index('caseobs_id')
            return ds_caseobs_data, df_caseobs_info
 
        # case 2: caseobs data save in CaseObsFolder_data
        Path_to_DS = {}
        for i in set_list:
            # ds set path
            ds_data_path = os.path.join(CaseObsFolder_data, i)
            
            # fetch ds
            try:
                ds = datasets.Dataset.load_from_disk(ds_data_path)
            except:
                logger.warning(f'\nError in the folder, remove it and skip it: {ds_data_path}')
                os.remove(ds_data_path); continue 
            
            # fetch caseobs_id_from_disk from ds
            caseobs_id_from_disk = set(ds.select_columns(['caseobs_id']).to_pandas()['caseobs_id'].unique())
                
            # check whether there are overlap between caseobs_ids and caseobs_id_from_disk, if these are no overlap, skip it. 
            if caseobs_ids is not None:
                overlap = caseobs_id_from_disk.intersection(caseobs_ids)
                if len(overlap) == 0: continue

            # append it to ds_data_path.
            Path_to_DS[ds_data_path] = ds
            
        if len(Path_to_DS) == 0:
            ds_caseobs_data = None 
            df_caseobs_info = pd.DataFrame(columns = ['caseobs_id', 'caseobs_idx_in_data']).set_index('caseobs_id')
            return ds_caseobs_data, df_caseobs_info
    
        # concatenate the datasets and save to the disk
        ds_caseobs_data = datasets.concatenate_datasets([v for _, v in Path_to_DS.items()])
        df_caseobs_info = ds_caseobs_data.select_columns(['caseobs_id']).to_pandas().reset_index().set_index('caseobs_id').rename(columns = {'index': 'caseobs_idx_in_data'})
        return ds_caseobs_data, df_caseobs_info

    def save_new_caseobs_to_ds_caseobs(self):

        new_calculated_caseobs = self.new_calculated_caseobs
        if len(new_calculated_caseobs) == 0:
            logger.info('No new calculated caseobs'); return None 

        # get the new data and information
        ds_caseobs_data_new = datasets.Dataset.from_pandas(pd.DataFrame([add_key_return_dict(v, 'caseobs_id', k) 
                                                                         for k, v  in new_calculated_caseobs.items()]))

        # we only save the new caseobs
        time.sleep(np.random.rand())
        time_finger = datetime.now().isoformat()[:-4].replace(':', '-') + str(round(np.random.rand()* 100, 0))
        new_path = os.path.join(self.CaseObsFolder_data, f'set-dt{time_finger}-sz{len(ds_caseobs_data_new)}')
        ds_caseobs_data_new.save_to_disk(new_path)
        
    def clean_and_update_ds_caseobs_data_from_disk(self, CaseObsFolder_data):
        if not os.path.exists(CaseObsFolder_data): return None   
        
        # get the set_list
        set_list = os.listdir(CaseObsFolder_data)
        if len(set_list) == 0: return None

        Path_to_SmallDS = {}
        for i in set_list:
            # ds set path
            ds_data_path = os.path.join(CaseObsFolder_data, i)
            
            # fetch ds
            try:
                ds = datasets.Dataset.load_from_disk(ds_data_path)
            except:
                logger.warning(f'\nError in the folder, remove it and skip it: {ds_data_path}')
                os.remove(ds_data_path); continue 
               
            if len(ds) >= 1000000: continue
            # check whether there are overlap between caseobs_ids and caseobs_id_from_disk, if these are no overlap, skip it. 
            Path_to_SmallDS[ds_data_path] = ds

        # ----------------------------------- TO UPDATE: the new version in case df_caseobs_not that big -----------------------------------
        # update the ds_caseobs_data into the disk
        if len(Path_to_SmallDS) == 1: return None
        ds_caseobs_data = concatenate_datasets([v for _, v in Path_to_SmallDS.items()])
        time.sleep(np.random.rand())
        time_finger = datetime.now().isoformat()[:-4].replace(':', '-') + str(round(np.random.rand()* 100, 0))
        new_path = os.path.join(self.CaseObsFolder_data, f'set-dt{time_finger}-sz{len(ds_caseobs_data)}')
        logger.info(f'save a group of small ds as a big ds to new_path: {new_path}')
        ds_caseobs_data.save_to_disk(new_path)

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
        
