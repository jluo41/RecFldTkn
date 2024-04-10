import os
import sys 
import logging
# import pickle 
# import pandas as pd
# from datetime import datetime 
# from pprint import pprint 

# import datasets
# from datasets.fingerprint import Hasher

# WorkSpace
KEY = 'WorkSpace'; WORKSPACE_PATH = os.getcwd().split(KEY)[0] + KEY; print(WORKSPACE_PATH)
os.chdir(WORKSPACE_PATH)
sys.path.append(WORKSPACE_PATH)

# Pipeline Space
from proj_space import SPACE
SPACE['WORKSPACE_PATH'] = WORKSPACE_PATH
sys.path.append(SPACE['CODE_FN'])

# Available Packages

# from recfldtkn.pipeline_dataset import process_df_tagging_tasks_in_chunks

# from recfldtkn.loadtools import load_ds_rec_and_info
# from recfldtkn.configfn import load_cohort_args, load_record_args
# from recfldtkn.loadtools import fetch_trigger_tools

# from recfldtkn.observer import get_CaseFeatInfo_for_a_CaseFeatName
# from recfldtkn.observer import CaseFeatureTransformer
# from recfldtkn.observer import get_fn_case_GammaFullInfo

# from recfldtkn.pipeline_case import get_ds_case_to_process
# from recfldtkn.pipeline_case import process_df_casefeat_tasks_in_chunks
# from recfldtkn.pipeline_case import process_df_tagging_tasks_in_chunks
# from recfldtkn.pipeline_case import process_df_filtering_tasks
# from recfldtkn.pipeline_case import assign_caseSplitTag_to_dsCase
# from recfldtkn.pipeline_case import get_dfset_from_SetName

from recfldtkn.configfn import load_cohort_args
from config_observer.CF import cf_to_CaseFeatConfig
from config_observer.QCF import cf_to_QueryCaseFeatConfig
from config_observer.CKPD import ckpd_to_CkpdObsConfig
from recfldtkn.pipeline_dataset import pipeline_to_generate_dfcase_and_dataset



logger = logging.getLogger(__name__)
recfldtkn_config_path = os.path.join(SPACE['CODE_FN'], 'config_recfldtkn/')
cohort_args = load_cohort_args(recfldtkn_config_path, SPACE)
cohort_args['ckpd_to_CkpdObsConfig'] = ckpd_to_CkpdObsConfig
cohort_args['ObsDTName'] = 'ObsDT'
cohort_args['PID_ObsDT_columns'] = [cohort_args['RootID'], cohort_args['ObsDTName']]



CASE_TAGGING_PROC_CONFIG = {
    'use_CF_from_disk': False,
    'use_CO_from_disk': False,
    'start_chunk_id': 0,
    'end_chunk_id': None,
    'chunk_size': 500000,
    'save_to_pickle': True,
    'num_processors': 4,
}


CASE_FIEDLING_PROC_CONFIG = {
    'use_CF_from_disk': True,
    'use_CO_from_disk': True,
    'start_chunk_id': 0,
    'end_chunk_id': None,
    'chunk_size': 100000,
    'save_to_pickle': False,
    'num_processors': 4
}


SAVE_DF_CASE = True
SAVE_DS_DATA = True



if __name__ == '__main__':

    ################################ dataset 1: TrulicityRx Tagging ################################
    # 0. ************ RFT config ************
    # RecName_to_dsRec, RecName_to_dsRecInfo = {}, {}
    # cohort_label_list = [1]
    # # 1. ************ Case Trigger config ************
    # TriggerCaseMethod = 'AnyInv'
    # # 2. ************ InputCaseSetName ************
    # InputCaseSetName = None 
    # # 3. ************ CaseTagging: TagMethod_List ************
    # TagMethod_List = ['PttBasicDF', 'EgmBf1Y', 'InvEgmAf1W']
    # # 4. ************ CaseFiltering: FilterMethod_List ************
    # FilterMethod_List = [] 
    # # 5. ************ CaseSpliting: SplitDict ************
    # SplitDict = {} 
    # # 6. ************ CaseSet Selection ************
    # CaseSplitConfig = {}
    # # 7. ************ CaseFeat: Feature Enriching ************
    # CaseFeat_List = [] # ['RxLevel.DemoInvRx.ObsPnt', 'RxLevel.Egm.Af1W']


    # ################################ dataset 1: TrulicityRx Filtering and Split ################################
    # # # 0. ************ RFT config ************
    # RecName_to_dsRec, RecName_to_dsRecInfo = {}, {}
    # cohort_label_list = [1]
    # # # 1. ************ Case Trigger config ************
    # TriggerCaseMethod = 'AnyInv'
    # # # 2. ************ InputCaseSetName ************
    # InputCaseSetName = 'C1-AnyInv-PttBasicDF.EgmBf1Y.InvEgmAf1W' 
    # # # 3. ************ CaseTagging: TagMethod_List ************
    # TagMethod_List = []
    # # # 4. ************ CaseFiltering: FilterMethod_List ************
    # FilterMethod_List = ['fPttBasicDF', 'fTailObsDT']
    # # # 5. ************ CaseSpliting: SplitDict ************
    # SplitDict = {'RANDOM_SEED': 42, 'downsample_ratio': 1, 'out_ratio': 0.0, 'test_ratio': 0.2, 'valid_ratio': 0.0}
    # # # 6. ************ CaseSet Selection ************
    # CaseSplitConfig = {}
    # # # 7. ************ CaseFeat: Feature Enriching ************
    # CaseFeat_List = [] # ['RxLevel.DemoInvRx.ObsPnt', 'RxLevel.Egm.Af1W']


    # ################################ dataset 3: TrulicityRx Case Fielding ################################
    # 0. ************ RFT config ************
    RecName_to_dsRec, RecName_to_dsRecInfo = {}, {}
    cohort_label_list = [1]
    # 1. ************ Case Trigger config ************
    TriggerCaseMethod = 'AnyInv'
    # 2. ************ InputCaseSetName ************
    InputCaseSetName = 'C1-AnyInv-PttBasicDF.EgmBf1Y.InvEgmAf1W-fPttBasicDF.fTailObsDT-rs42.ds1.out0.0ts0.2vd0.0' 
    # 3. ************ CaseTagging: TagMethod_List ************
    TagMethod_List = []
    # 4. ************ CaseFiltering: FilterMethod_List ************
    FilterMethod_List = []
    # 5. ************ CaseSpliting: SplitDict ************
    SplitDict = {}
    # 6. ************ CaseSet Selection ************
    CaseSplitConfig = {
        'TrainSetName': 'In-Train_all',
        'EvalSetNames': ['In-Test_all']
    }
    # 7. ************ CaseFeat: Feature Enriching ************
    CaseFeat_List = ['InvLevel.DemoInvRx.ObsPnt', 'InvLevel.Egm.Af1W']


    results = pipeline_to_generate_dfcase_and_dataset(RecName_to_dsRec, 
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
                                                      SAVE_DF_CASE,
                                                      SAVE_DS_DATA,
                                                    )

    