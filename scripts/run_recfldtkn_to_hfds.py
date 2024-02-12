import os
import sys
import logging
from pprint import pprint 

KEY = 'WorkSpace'; WORKSPACE_PATH = os.getcwd().split(KEY)[0] + KEY
print(WORKSPACE_PATH); os.chdir(WORKSPACE_PATH)
sys.path.append(WORKSPACE_PATH)

from proj_space import PROJECT, TaskName, SPACE
sys.path.append(SPACE['CODE_FN'])

import argparse
import pandas as pd 
from datetime import datetime 
from recfldtkn.configfn import load_cohort_args, load_record_args, load_fldtkn_args
from recfldtkn.loadtools import load_module_variables, update_args_to_list, load_ds_rec_and_info, filter_with_cohort_label
from recfldtkn.pipeline_record import pipeline_record


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s:%(asctime)s:(%(filename)s@%(lineno)d %(name)s)]: %(message)s')

recfldtkn_config_path = os.path.join(SPACE['CODE_FN'], 'config_recfldtkn/')

my_parser = argparse.ArgumentParser(description='Process Input.')

# Add the arguments
my_parser.add_argument('--cohort_label',
                    metavar='cohort_label',
                    default = None, 
                    type=str,
                    help='the cohort_label to process')

# Add the arguments
my_parser.add_argument('--record_name',
                    metavar='record_name',
                    default = None, 
                    type=str,
                    help='the record_name to process')

# Add the arguments
my_parser.add_argument('--fldtkn_name_list',
                    metavar='fldtkn_name_list',
                    default = None, 
                    type=str,
                    nargs='+',  # This allows multiple values
                    help='the fldtkn_name_list to process')

if __name__ == '__main__':


    '''
    python ../scripts/run_recfldtkn_to_hfds.py \
        --cohort_label 1 \
        --record_name P \
        --fldtkn_name_list P-DemoTkn P-Zip3DemoNumeTkn P-Zip3EconNumeTkn P-Zip3HousingNumeTkn P-Zip3SocialNumeTkn
    
    python ../scripts/run_recfldtkn_to_hfds.py \
        --cohort_label 1 \
        --record_name PInv \
        --fldtkn_name_list PInv-InfoTkn

    python ../scripts/run_recfldtkn_to_hfds.py  \
        --cohort_label 1 \
        --record_name \
            "Rx" \
        --fldtkn_name_list \
            "Rx-CmpCateTkn" \
            "Rx-InsCateTkn" \
            "Rx-QuantNumeTkn" \
            "Rx-ServiceCateTkn" \
            "Rx-SysCateTkn"

    python ../scripts/run_recfldtkn_to_hfds.py --cohort_label 1  --record_name "EgmClick"
    python ../scripts/run_recfldtkn_to_hfds.py --cohort_label 1  --record_name "EgmAuthen"
    python ../scripts/run_recfldtkn_to_hfds.py --cohort_label 1  --record_name "EgmCallPharm"
    python ../scripts/run_recfldtkn_to_hfds.py --cohort_label 1  --record_name "EgmCopay"
    python ../scripts/run_recfldtkn_to_hfds.py --cohort_label 1  --record_name "EgmEdu"
    python ../scripts/run_recfldtkn_to_hfds.py --cohort_label 1  --record_name "EgmRmd"
    '''

    args = my_parser.parse_args()
    RecName = args.record_name # 'CGM5Min'

    # cohort_args
    cohort_args = load_cohort_args(recfldtkn_config_path, SPACE)
    RootIDLength = cohort_args['RootIDLength']
    RootID = cohort_args['RootID']

    cohort_args = load_cohort_args(recfldtkn_config_path, SPACE)
    RootIDLength = cohort_args['RootIDLength']
    RootID = cohort_args['RootID']

    # rec_folder = cohort_args['rec_folder']
    RawRootID = cohort_args['RawRootID'] 
    record_args = load_record_args(RecName, cohort_args)

    # cohort_label and cohort_name
    cohort_label = int(args.cohort_label)
    cohort_config = [v for k, v in cohort_args['CohortInfo'].items() if v['cohort_label'] == cohort_label][0]
    cohort_name = cohort_config['cohort_name']

    # RecName and FldTknName_List
    RecName = args.record_name # 'CGM5Min'
    FldTknName_List = update_args_to_list(args.fldtkn_name_list) 
    if args.fldtkn_name_list is not None:
        FldTknName_List = [args.fldtkn_name_list] if type(args.fldtkn_name_list) != list else args.fldtkn_name_list #  ['CGM5Min-N2Cin1Tkn']
    else:
        FldTknName_List = []

    ######################### prepare your input data. 
    record_to_recfldtkn_list = {RecName: FldTknName_List}
    logger.info(f'record_to_recfldtkn_list: {record_to_recfldtkn_list}')
    #########################

    cohort_label_list = [cohort_label]
    # print(cohort_label_list)
    ds_Human, _ = load_ds_rec_and_info(cohort_args['RecName'], cohort_args, cohort_label_list = cohort_label_list)
    df_Human = ds_Human.to_pandas()

    results = pipeline_record(record_to_recfldtkn_list, 
                              cohort_name, cohort_label,  
                              df_Human, cohort_args, 
                              load_from_disk = False, 
                              reuse_old_rft = False, 
                              save_to_disk = True)
    
    record_to_ds_rec = results['record_to_ds_rec']
    
    for record, ds_rec in record_to_ds_rec.items():
        logger.info(f'{record}: {ds_rec.column_names}')
        logger.info(ds_rec)

    