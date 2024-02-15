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
    GROUP_NUM = 12
    downsample_ratio = float(1.0)
    case_id_columns = ['PID', 'ObsDT', 'a1cV0', 'a1cV1', 
                       # 'DT', 'ECID', 
                       'birthdate',
                       'ethnicity', 'language', 'zipcode',
                       'Sex', 'Race', 'CohortLabel', 'Age',
                        ]
                        
    ####################
    
    args = my_parser.parse_args()
    if args.run_case_obs:
        
        
        if args.task_name == 'ObserveCOs':
            ########## <-------- change this
            # case_type_list = ['PDA1C'] # 180k
            # case_observations = [
            #     'BfA1CInfo:ro.A1C-Bf10Y_ct.BfA1CInfo',  
            # ]
            
            
            ########## <-------- change this
            # case_type_list = ['BfA1CNoDorPD'] # 65k
            # case_observations = [
            #     'BfDiagDBInfo:ro.Diag-Bf10Y_ct.BfDiagDBInfo',  
            #     'BfMedDBInfo:ro.Med-Bf10Y_ct.BfMedDBInfo',
            # ]
            
            
            ########## <-------- change this
            # case_type_list = ['BfNoDorPD'] # 65k
            # case_observations = [
            #     'AfA1CInfo:ro.A1C-Af1Y_ct.AfA1CInfo', 
            #     'AfA1CPntInfo:ro.A1C-PntAf1Y3M_ct.AfA1CInfo',
            #     'AfDiagDBInfo:ro.Diag-Af1Y_ct.AfDiagDBInfo',  
            #     'AfMedDBInfo:ro.Med-Af1Y_ct.AfMedDBInfo',
            # ]
            
            
            ########## <-------- change this
            case_type_list = ['BfNoDorPD'] # 65k
            case_observations = [
                'CO1:ro.A1C-Bf1Y_ct.RecNum', 
                'CO2:ro.BMI-Bf1Y_ct.RecNum', 
                'CO3:ro.ALT-Bf1Y_ct.RecNum', 
                'CO4:ro.HDL-Bf1Y_ct.RecNum', 
                'CO5:ro.LDL-Bf1Y_ct.RecNum', 
                'CO6:ro.DBP-Bf1Y_ct.RecNum',
                'CO7:ro.SBP-Bf1Y_ct.RecNum',
                'CO8:ro.Diag-Bf1Y_ct.RecNum',
                'CO9:ro.Med-Bf1Y_ct.RecNum',
                'CO10:ro.Alcohol-Bf1Y_ct.RecNum',
                'CO11:ro.Exercise-Bf1Y_ct.RecNum',
                'CO12:ro.Diet-Bf1Y_ct.RecNum',
                'CO13:ro.Smoking-Bf1Y_ct.RecNum',
                'CO14:ro.NoteType-Bf1Y_ct.RecNum',
                'CO15:ro.PnSect-Bf1Y_ct.RecNum',
            ]
            
            ##########
            '''
            python bash_ds.py --group_id_list 0    --run_case_obs --task_name ObserveCOs
            python bash_ds.py --group_id_list all  --run_case_obs --task_name ObserveCOs
            '''
        else:
            raise NotImplementedError
        
        for case_type in case_type_list:
            # main_case_observation(args, case_type, case_observations, case_id_columns)
            bash_run_case_observation(args, case_type, case_observations, case_id_columns, batch_size, GROUP_NUM) 

    if args.run_case_taskop:
        
        if args.task_name == 'BfA1CNoDorPD':
            ########## <-------- change this
            case_type_list = ['PDA1C']
            case_observations = [
                'BfA1CInfo:ro.A1C-Bf10Y_ct.BfA1CInfo',  
            ]
            case_taskop = 'BfA1CNoDorPD'
            post_process = 'filter'
            ##########
            '''
            python bash_ds.py --group_id_list 0    --run_case_taskop --task_name BfA1CNoDorPD
            python bash_ds.py --group_id_list all  --run_case_taskop --task_name BfA1CNoDorPD
            '''
            
        elif args.task_name == 'BfNoDorPD':
            ########## <-------- change this
            case_type_list = ['BfA1CNoDorPD']
            case_observations = [
                'BfA1CInfo:ro.A1C-Bf10Y_ct.BfA1CInfo', 
                'BfDiagDBInfo:ro.Diag-Bf10Y_ct.BfDiagDBInfo', 
                'BfMedDBInfo:ro.Med-Bf10Y_ct.BfMedDBInfo', 
            ]
            case_taskop = 'BfNoDorPD'
            post_process = 'filter'
            ##########
            '''
            python bash_ds.py --group_id_list 0    --run_case_taskop --task_name BfNoDorPD
            python bash_ds.py --group_id_list all  --run_case_taskop --task_name BfNoDorPD
            '''
            
        elif args.task_name == 'AfDiabLabel':
            ########## <-------- change this
            case_type_list = ['BfNoDorPD']
            case_observations = [
                'AfA1CInfo:ro.A1C-Af1Y_ct.AfA1CInfo', 
                'AfDiagDBInfo:ro.Diag-Af1Y_ct.AfDiagDBInfo', 
                'AfMedDBInfo:ro.Med-Bf1Y_ct.AfMedDBInfo', 
                
                'AfA1CPntInfo:ro.A1C-PntAf1Y2M_ct.AfA1CInfo', 
            ]
            case_taskop = 'AfDiabLabel'
            post_process = 'filter'
            ##########
            '''
            python bash_ds.py --group_id_list 0    --run_case_taskop --task_name AfDiabLabel
            python bash_ds.py --group_id_list all  --run_case_taskop --task_name AfDiabLabel
            '''
            
        elif args.task_name == 'EduRxMLPred': # <--- machine learning version.
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
            ##########
            '''
            # python bash_ds.py --group_id_list all  --core_num 1 --run_case_taskop --task_name EduRxMLPred
            # python bash_ds.py --group_id_list 4  --core_num 1 --run_case_taskop --task_name EduRxMLPred
            '''
        else:
            raise NotImplementedError
        
        bash_run_case_taskop(args, case_type_list, case_observations, case_id_columns, case_taskop, post_process, batch_size, GROUP_NUM)
        