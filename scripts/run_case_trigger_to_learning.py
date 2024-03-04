import os
import sys 
import logging
import pandas as pd 
from pprint import pprint 

# WorkSpace
KEY = 'WorkSpace'; WORKSPACE_PATH = os.getcwd().split(KEY)[0] + KEY; print(WORKSPACE_PATH)
os.chdir(WORKSPACE_PATH)
sys.path.append(WORKSPACE_PATH)

# Pipeline Space
from proj_space import SPACE
SPACE['WORKSPACE_PATH'] = WORKSPACE_PATH
sys.path.append(SPACE['CODE_FN'])
pprint(SPACE)

# Available Packages
import pandas as pd
from datetime import datetime 

logger = logging.getLogger(__name__)
recfldtkn_config_path = os.path.join(SPACE['CODE_FN'], 'config_recfldtkn/')


import datasets
from recfldtkn.loadtools import load_ds_rec_and_info
from recfldtkn.configfn import load_cohort_args, load_record_args

base_config = load_cohort_args(recfldtkn_config_path, SPACE)
print(base_config)

from recfldtkn.pipeline_model import get_Trigger_Cases, convert_TriggerCases_to_LearningCases
from recfldtkn.pipeline_model import split_dataframe


use_learning = True 
use_inference = not use_learning

######################################
TriggerCaseMethod = 'CGM5MinEntry'
cohort_label_list = [1]
Trigger2LearningMethods = [
    
    {'op':'Tag',    'Name': 'Tag_PttBasicWD'},
    {'op':'Filter', 'Name': 'Filter_PttBasicWD'},
    
    {'op':'CFQ',    'Name': 'CFQ_Bf24hCGMrn'},
    {'op':'TagCF',  'Name': 'TagCF_Bf24hCGMrn', 'CFQName': 'CFQ_Bf24hCGMrn'},
    
    {'op':'CFQ',    'Name': 'CFQ_Af2hCGMrn',    
     'type': 'learning-only'},
    {'op':'TagCF',  'Name': 'TagCF_Af2hCGMrn',  'CFQName': 'CFQ_Af2hCGMrn',  
     'type': 'learning-only',},
    
    # {'op':'Filter', 'Name': 'Filter_BfCGMgeq280'},
    # {'op':'Filter', 'Name': 'Filter_AfCGMgeq24',  
    #  'type': 'learning-only'},
    
    # {'op':'CFQ',    'Name': 'CFQ_Bf1mMEDALrn',    
    #  'type': 'learning-only'},
    # {'op':'TagCF',  'Name': 'TagCF_Bf1mMEDALrn', 'CFQName': 'CFQ_Bf1mMEDALrn', 
    #  'type': 'learning-only'},

]
######################################

RecName_to_dsRec = {}
RecName_to_dsRecInfo = {}

df_case = get_Trigger_Cases(TriggerCaseMethod, 
                            cohort_label_list, 
                            base_config, 
                            SPACE, 
                            RecName_to_dsRec, 
                            RecName_to_dsRecInfo)




SIZE = 1_000_000
chunks = split_dataframe(df_case, SIZE)



logger.info(f'Number of chunks: {len(chunks)}')



chuck_id = 10

df_case = chunks[chuck_id]

df_case = convert_TriggerCases_to_LearningCases(df_case, 
                                                cohort_label_list, 
                                                Trigger2LearningMethods, 
                                                base_config, 
                                                use_inference)
columns = [i for i in df_case.columns if '_co.' not in i]
# print(columns)
df_case = df_case[columns].reset_index(drop=True)
        