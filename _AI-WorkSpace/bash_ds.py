import os
import sys 
import logging
import random
import pandas as pd 
from pprint import pprint 
from IPython.display import display, HTML

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
import pandas as pd
from recfldtkn.bash_utils import bash_run_case_observation, bash_run_case_taskop

logger = logging.getLogger(__name__)

##################################
my_parser = argparse.ArgumentParser(description='Process Input.')
my_parser.add_argument('--run_case_obs', action='store_true', help='Enable step1')
my_parser.add_argument('--run_case_taskop', action='store_true', help='Enable step1')
my_parser.add_argument('--task_name', default = None, type = str)
my_parser.add_argument('--group_id_list', nargs='+')
my_parser.add_argument('--case_type', default = None, type = str)
my_parser.add_argument('--case_id_columns', nargs='+')
my_parser.add_argument('--core_num', default=1, type=int)
my_parser.add_argument('--post_process', default = None, type = str)


if __name__ == "__main__":
    ####################
    batch_size = 1000 
    random_seed = 42
    GROUP_NUM = 15
    downsample_ratio = float(1.0)
    case_id_columns = ['PID', 'ObsDT', 'PInvID', 'RxID']
    ####################
    
    args = my_parser.parse_args()
    if args.run_case_obs:
        if args.task_name == 'ObserveCOs':
            ########## <-------- change this
            case_type_list = ['whole']
            case_observations = [
                'FutEduTknY:ro.EgmEdu-Af1W_ct.FutRxEduTkn',  
            ]
            ##########
            '''
            python bash.py --group_id_list all  --core_num 1 --run_case_obs --task_name ObserveCOs
            '''
        else:
            raise NotImplementedError
        
        for case_type in case_type_list:
            # main_case_observation(args, case_type, case_observations, case_id_columns)
            bash_run_case_observation(args, case_type, case_observations, case_id_columns, batch_size, GROUP_NUM) 

    if args.run_case_taskop:
        if args.task_name == 'EduRxMLPred': # <--- machine learning version.
            ########## <-------- change this
            case_type_list = [ # after the a-process. 
                'TrulicityRx-rs42-ds1-out0ts2023.11.01vd0.1-in_train',
                'TrulicityRx-rs42-ds1-out0ts2023.11.01vd0.1-in_valid',
                'TrulicityRx-rs42-ds1-out0ts2023.11.01vd0.1-in_test',
            ]

            # and all subgroups.

            case_observations = [
                'FutEduTknY:ro.EgmEdu-Af1W_ct.FutRxEduTkn',  
                'PDemo:ro.P-Demo_ct.InCaseTkn',
                'PZip3Demo:ro.P-Zip3DemoNume_ct.InCaseTkn',
                'PZip3Econ:ro.P-Zip3EconNume_ct.InCaseTkn',
                'PZip3House:ro.P-Zip3HousingNume_ct.InCaseTkn',
                'PZip3Social:ro.P-Zip3SocialNume_ct.InCaseTkn',
                'RxInCase1:ro.Rx-InObs-CmpCate_ct.InCaseTkn',
                'RxInCase2:ro.Rx-InObs-InsCate_ct.InCaseTkn',
                'RxInCase3:ro.Rx-InObs-QuantN2C_ct.InCaseTkn',
                'RxInCase4:ro.Rx-InObs-QuantNume_ct.InCaseTkn',
                'RxInCase5:ro.Rx-InObs-ServiceCate_ct.InCaseTkn',
                'RxInCase6:ro.Rx-InObs-SysCate_ct.InCaseTkn',
                'RxInObsNum:ro.Rx-InObs_ct.RecNum',
            ]


            case_taskop = 'EduRxMLPred'

            post_process = 'aidataset'


            CO_list_hash = hash(tuple(sorted(case_observations)))
            print(CO_list_hash)
            ##########
            '''
            # python bash_ds.py --group_id_list all  --core_num 1 --run_case_taskop --task_name EduRxMLPred
            # python bash_ds.py --group_id_list 4  --core_num 1 --run_case_taskop --task_name EduRxMLPred
            '''
        else:
            raise NotImplementedError
        
        bash_run_case_taskop(args, case_type_list, case_observations, case_id_columns, case_taskop, post_process, batch_size, GROUP_NUM)
        