import os
import sys 
import logging
import argparse

# WorkSpace
KEY = 'WorkSpace'; WORKSPACE_PATH = os.getcwd().split(KEY)[0] + KEY; print(WORKSPACE_PATH)
os.chdir(WORKSPACE_PATH)
sys.path.append(WORKSPACE_PATH)

# Pipeline Space
from proj_space import SPACE
SPACE['WORKSPACE_PATH'] = WORKSPACE_PATH
sys.path.append(SPACE['CODE_FN'])

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
    'num_processors': 8, # 12,
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
LOAD_DF_CASE = True
LOAD_DS_DATA = True
SAVE_TRIGGER_DF = True # False
RANDOM_SAMPLE = None # 10000 

parser = argparse.ArgumentParser(description="Run data processing pipelines with configurable settings")
parser.add_argument('--ds_config_name', type=str, default='tagBfAfCGMrn', help='Path to the configuration file')



# OutDataset = 'xxx'

if __name__ == '__main__':

    args = parser.parse_args()
    ds_config_name = args.ds_config_name

    if ds_config_name == 'tagBfAfCGMinfo':
        # ################################################################
        # 0. ************ RFT config ************
        RecName_to_dsRec, RecName_to_dsRecInfo = {}, {}
        cohort_label_list = [1, 2, 3]
        # 1. ************ Case Trigger config ************
        TriggerCaseMethod = 'CGM5MinEntry'
        # 2. ************ InputCaseSetName ************
        InputCaseSetName = None
        # 3. ************ CaseTagging: TagMethod_List ************
        TagMethod_List = ['Bf24hCGMinfo', 'Af2hCGMinfo']
        # 4. ************ CaseFiltering: FilterMethod_List ************
        FilterMethod_List = []
        # 5. ************ CaseSpliting: SplitDict ************
        SplitDict = {}
        # 6. ************ CaseSet Selection ************
        CaseSplitConfig = {}
        # 7. ************ CaseFeat: Feature Enriching ************
        CaseFeat_List = []

    elif ds_config_name == 'filterBfAfCGMrn':
        # ################################################################
        # 0. ************ RFT config ************
        RecName_to_dsRec, RecName_to_dsRecInfo = {}, {}
        cohort_label_list = [1, 2, 3]
        # 1. ************ Case Trigger config ************
        TriggerCaseMethod = 'CGM5MinEntry'
        # 2. ************ InputCaseSetName ************
        InputCaseSetName = 'C1.2.3-CGM5MinEntry-sz21215912-Bf24hCGMinfo.Af2hCGMinfo'
        # 3. ************ CaseTagging: TagMethod_List ************
        TagMethod_List = []
        # 4. ************ CaseFiltering: FilterMethod_List ************
        FilterMethod_List = ['fBf24h280CGM', 'fAf2h24CGM', 'fBf24HModePctn40']
        # FilterMethod_List = ['fBf24h289CGM', 'fAf2h24CGM']
        # 5. ************ CaseSpliting: SplitDict ************
        SplitDict = {}
        # 6. ************ CaseSet Selection ************
        CaseSplitConfig = {}
        # 7. ************ CaseFeat: Feature Enriching ************
        CaseFeat_List = []


    elif ds_config_name == 'tagBfAfCGMinfo':
        # ################################################################
        # 0. ************ RFT config ************
        RecName_to_dsRec, RecName_to_dsRecInfo = {}, {}
        cohort_label_list = [1, 2, 3]
        # 1. ************ Case Trigger config ************
        TriggerCaseMethod = 'CGM5MinEntry'
        # 2. ************ InputCaseSetName ************
        InputCaseSetName = 'C1.2.3-CGM5MinEntry-sz21215912-Bf24hCGMrn.Af2hCGMrn-fBf24h280CGM.fAf2h24CGM'
        # 3. ************ CaseTagging: TagMethod_List ************
        TagMethod_List = ['Bf24hCGMinfo', 'Af2hCGMinfo']
        # 4. ************ CaseFiltering: FilterMethod_List ************
        FilterMethod_List = []
        # FilterMethod_List = ['fBf24h289CGM', 'fAf2h24CGM']
        # 5. ************ CaseSpliting: SplitDict ************
        SplitDict = {}
        # 6. ************ CaseSet Selection ************
        CaseSplitConfig = {}
        # 7. ************ CaseFeat: Feature Enriching ************
        CaseFeat_List = []

    else:
        raise ValueError(f"Invalid ds_config_name: {ds_config_name}")



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
                                                      LOAD_DF_CASE, 
                                                      LOAD_DS_DATA,
                                                      RANDOM_SAMPLE,
                                                      SAVE_TRIGGER_DF,
                                                    )

    