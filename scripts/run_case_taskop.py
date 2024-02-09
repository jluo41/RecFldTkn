import os
import sys 
import logging
import random
import pandas as pd 
from pprint import pprint 
from IPython.display import display, HTML
import shutil

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
import argparse
import datasets
import tokenizers
import pandas as pd
from datetime import datetime 

from recfldtkn.ckpd_obs import Ckpd_ObservationS
from recfldtkn.configfn import load_cohort_args
from recfldtkn.obsname import convert_RecObsName_and_CaseTkn_to_CaseObsName
from recfldtkn.loadtools import load_module_variables, update_args_to_list
from recfldtkn.observer import get_caseset_to_observe
from recfldtkn.constaidatatools import convert_case_observations_to_co_to_observation
from recfldtkn.constaidatatools import get_RecNameList_and_FldTknList
from recfldtkn.constaidatatools import pipeline_to_generate_co_to_CaseObsInfo, get_complete_dataset
from recfldtkn.constaidatatools import create_tokenizer
from recfldtkn.constaidatatools import pipeline_for_ai_dataset_construction

from datasets import disable_caching
from tokenizers.pre_tokenizers import WhitespaceSplit

disable_caching()

logger = logging.getLogger(__name__)
recfldtkn_config_path = os.path.join(SPACE['CODE_RFT'], 'config_recfldtkn/')


##################################
my_parser = argparse.ArgumentParser(description='Process Input.')

# Add the arguments
my_parser.add_argument('--group_id_list',
                    metavar='group_id_list',
                    nargs='+',  
                    default = None, 
                    type=str)

my_parser.add_argument('--case_type_list',
                    metavar='case_type_list',
                    nargs='+',  
                    default = None, 
                    type=str)

my_parser.add_argument('--case_id_columns',
                    metavar='case_id_columns',
                    nargs='+',  
                    default = None, 
                    type=str)

my_parser.add_argument('--case_observations',
                    metavar='case_observations',
                    nargs='+',  
                    default = None, 
                    type=str)


my_parser.add_argument('--case_taskop',
                    metavar='case_taskop',
                    # nargs='+',  
                    default = None, 
                    type=str)

my_parser.add_argument('--post_process',
                        metavar='case_taskop',
                        default = 'type', # 'datapoint'
                        type=str)

my_parser.add_argument('--batch_size', 
                        default = 1000,
                        type=int)

if __name__ == '__main__':

    # step 1: get the arguments
    args = my_parser.parse_args()
    case_observations = args.case_observations
    case_type_list = update_args_to_list(args.case_type_list)
    group_id_list = update_args_to_list(args.group_id_list)
    if 'all' in group_id_list: 
        group_num = len(os.listdir(os.path.join(SPACE['DATA_TASK'], 'CaseFolder', 'whole')))
        group_id_list = list(range(group_num))
    else:
        group_id_list = [int(i) for i in update_args_to_list(args.group_id_list)]
    print('group_id_list: ', group_id_list)
    case_id_columns = update_args_to_list(args.case_id_columns) 
    CaseTaskOp = args.case_taskop

    # step 2: ds_case_dict: get the case to be observed
    d = {}
    for case_type in case_type_list:
        print('\n========================')
        print('case_type---->', case_type)
        # L = []
        for group_id in group_id_list:
            print('group_id------->', group_id)
            CaseFolder = os.path.join(SPACE['DATA_TASK'], 'CaseFolder', case_type)
            print(CaseFolder)
            cohort_args = load_cohort_args(recfldtkn_config_path, SPACE)
            cohort_args['Ckpd_ObservationS'] = Ckpd_ObservationS
            print(cohort_args)
            group_name, ds_case = get_caseset_to_observe(group_id, CaseFolder, case_id_columns, cohort_args)
            print(group_name)
            print(ds_case)
            # L.append(ds_case)
            # ds_case_type = datasets.concatenate_datasets(L)
            if ds_case is not None:
                d[case_type + '|' + group_name] = ds_case
    ds_case_dict = datasets.DatasetDict(d)
    print(ds_case_dict)

    
    print(f'\n================ Gamma: Case Task Operation-{CaseTaskOp} ================')
    results = pipeline_for_ai_dataset_construction(ds_case_dict, 
                                                    case_observations, 
                                                    CaseTaskOp,
                                                    case_id_columns, 
                                                    cohort_args, SPACE)
    CaseTaskOpVocab = results['CaseTaskOpVocab']
    ds_case_proc = results['ds_case_proc']
    
    # step 5: post process
    if args.post_process == 'filter':
        for case_split_type, ds_case in ds_case_proc.items():
            df = ds_case.select_columns(case_id_columns + ['tid']).to_pandas()
            df_filter = df[df['tid'].apply(lambda x: x[0]) != 'None'][case_id_columns].reset_index(drop=True)
            # print(case_split_type)
            case_type, group_name = case_split_type.split('|')
            path = os.path.join(SPACE['DATA_TASK'], 'CaseFolder', CaseTaskOp.lower(), group_name + '.p')
            if not os.path.exists(os.path.dirname(path)): os.makedirs(os.path.dirname(path))
            print(path)
            df_filter.to_pickle(path)
            print(df_filter.head())
            print(df_filter.shape)

    if args.post_process == 'aidataset':
        for case_split_type, ds_case in ds_case_proc.items():
            case_type, group_name = case_split_type.split('|')
            path = os.path.join(SPACE['MODEL_TASK'], CaseTaskOp, 'AIDataset', case_type, group_name)
            if not os.path.exists(os.path.dirname(path)): os.makedirs(os.path.dirname(path))
            print('generate ai dataset to: ', path)
            print(ds_case)
            ds_case.save_to_disk(path)
            ds_case = datasets.load_from_disk(path) # adding this as a check.
        
        # check the tokenizer
        tokenizer_folder = os.path.join(SPACE['MODEL_TASK'], CaseTaskOp)
        tokenizer = create_tokenizer(CaseTaskOpVocab, tokenizer_folder)

    else:
        print(f'args.post_process: "{args.post_process}" is not specifed')
        raise NotImplementedError

