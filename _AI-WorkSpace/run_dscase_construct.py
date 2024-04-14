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
    'num_processors': 1, # 12,
}


CASE_FIEDLING_PROC_CONFIG = {
    'use_CF_from_disk': False,
    'use_CO_from_disk': True,
    'start_chunk_id': 0,
    'end_chunk_id': None,
    'chunk_size': 100000,
    'save_to_pickle': False,
    'num_processors': 4,
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
        TagMethod_List = [
            'PttBasicWD', 
            'Bf24hCGMinfo', 'Af2hCGMinfo',
            # 'Bf24hCGMrn', 'Af2hCGMrn',
            # 'Bf24hCGMmode', 'Af2hCGMmode', 
            ]
        # 4. ************ CaseFiltering: FilterMethod_List ************
        FilterMethod_List = []
        # 5. ************ CaseSpliting: SplitDict ************
        SplitDict = {}
        # 6. ************ CaseSet Selection ************
        CaseSplitConfig = {}
        # 7. ************ CaseFeat: Feature Enriching ************
        CaseFeat_List = []

    elif ds_config_name == 'fltBfAfCGMinfo':
        # ################################################################
        # 0. ************ RFT config ************
        RecName_to_dsRec, RecName_to_dsRecInfo = {}, {}
        cohort_label_list = [1, 2, 3]
        # 1. ************ Case Trigger config ************
        TriggerCaseMethod = 'CGM5MinEntry'
        # 2. ************ InputCaseSetName ************
        InputCaseSetName = 'C1.2.3-CGM5MinEntry-sz21215912-tagBfAfCGMinfo'
        # 3. ************ CaseTagging: TagMethod_List ************
        TagMethod_List = []
        # 4. ************ CaseFiltering: FilterMethod_List ************
        FilterMethod_List = [
                             'fPttBasicWD',
                             'fBf24h280CGM', 'fAf2h24CGM', 
                             'fBf24HModePctn40', 'fAf2HModePctn40']
        # FilterMethod_List = ['fBf24h289CGM', 'fAf2h24CGM']
        # 5. ************ CaseSpliting: SplitDict ************
        SplitDict = {}
        # 6. ************ CaseSet Selection ************
        CaseSplitConfig = {}
        # 7. ************ CaseFeat: Feature Enriching ************
        CaseFeat_List = []

    elif ds_config_name == 'splitR42ds10':
        # ################################################################
        # 0. ************ RFT config ************
        RecName_to_dsRec, RecName_to_dsRecInfo = {}, {}
        cohort_label_list = [1, 2, 3]
        # 1. ************ Case Trigger config ************
        TriggerCaseMethod = 'CGM5MinEntry'
        # 2. ************ InputCaseSetName ************
        InputCaseSetName = 'C1.2.3-CGM5MinEntry-sz21215912-tagBfAfCGMinfo-fltBfAfCGMinfo'
        # 3. ************ CaseTagging: TagMethod_List ************
        TagMethod_List = []
        # 4. ************ CaseFiltering: FilterMethod_List ************
        FilterMethod_List = []
        # FilterMethod_List = ['fBf24h289CGM', 'fAf2h24CGM']
        # 5. ************ CaseSpliting: SplitDict ************
        SplitDict = {
            'RANDOM_SEED': 42,
            'downsample_ratio': 0.1,
            'out_ratio': 0.1,
            'test_ratio': 'tail0.1',
            'valid_ratio': 0.1
        }
        # 6. ************ CaseSet Selection ************
        CaseSplitConfig = {
            # 'TrainSetName': 'In_Train',
            # 'EvalSetNames': ['In_Valid', 'In_Test', 'Out_Valid', 'Out_Test', 'Out_Train'],
        }
        # 7. ************ CaseFeat: Feature Enriching ************
        CaseFeat_List = []
        
    elif ds_config_name == 'CgmGptData':
        # ################################################################
        # 0. ************ RFT config ************
        RecName_to_dsRec, RecName_to_dsRecInfo = {}, {}
        cohort_label_list = [1, 2, 3]
        # 1. ************ Case Trigger config ************
        TriggerCaseMethod = 'CGM5MinEntry'
        # 2. ************ InputCaseSetName ************
        InputCaseSetName = 'C1.2.3-CGM5MinEntry-sz21215912-tagBfAfCGMinfo-fltBfAfCGMinfo-splitR42ds10'
        # 3. ************ CaseTagging: TagMethod_List ************
        TagMethod_List = []
        # 4. ************ CaseFiltering: FilterMethod_List ************
        FilterMethod_List = []
        # FilterMethod_List = ['fBf24h289CGM', 'fAf2h24CGM']
        # 5. ************ CaseSpliting: SplitDict ************
        SplitDict = {}
        # 6. ************ CaseSet Selection ************
        CaseSplitConfig = {
            'TrainSetName': 'In-Train_all',
            'EvalSetNames': ['In-Valid_all', 'In-Test_all', 
                             'Out_all', 'Out-Test_all'],
        }
        # 7. ************ CaseFeat: Feature Enriching ************
        CaseFeat_List =  ['TargetCGM.Bf24H', 'TargetCGM.Af2H']
        # CaseFeat_List = []
        
        
    elif ds_config_name == 'CgmGptMedalData':
        # ################################################################
        # 0. ************ RFT config ************
        RecName_to_dsRec, RecName_to_dsRecInfo = {}, {}
        cohort_label_list = [1, 2, 3]
        # 1. ************ Case Trigger config ************
        TriggerCaseMethod = 'CGM5MinEntry'
        # 2. ************ InputCaseSetName ************
        InputCaseSetName = 'C1.2.3-CGM5MinEntry-sz21215912-tagBfAfCGMinfo-fltBfAfCGMinfo-splitR42ds10'
        # 3. ************ CaseTagging: TagMethod_List ************
        TagMethod_List = []
        # 4. ************ CaseFiltering: FilterMethod_List ************
        FilterMethod_List = []
        # FilterMethod_List = ['fBf24h289CGM', 'fAf2h24CGM']
        # 5. ************ CaseSpliting: SplitDict ************
        SplitDict = {}
        # 6. ************ CaseSet Selection ************
        CaseSplitConfig = {
            'TrainSetName': 'In-Train_all',
            'EvalSetNames': ['In-Valid_all', 'In-Test_all', 
                             'Out_all', 'Out-Test_all'],
        }
        # 7. ************ CaseFeat: Feature Enriching ************
        CaseFeat_List =   [
                            'TargetCGM.Bf24H', 
                            'TargetCGM.Af2H',
                            'Time.Bf24H', 
                            'Time.Af2H',
                            'Food.Bf24H', 
                            'Food.Af2H',
                            ]
        # CaseFeat_List = []

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
                                                      FinalCaseSetName_SUFFIX = ds_config_name,
                                                    )

    