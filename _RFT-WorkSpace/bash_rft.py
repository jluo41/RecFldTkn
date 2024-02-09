import os
import sys 
import logging
import argparse

# WorkSpace
KEY = 'WorkSpace'; 
WORKSPACE_PATH = os.getcwd().split(KEY)[0] + KEY; 
print(WORKSPACE_PATH)
os.chdir(WORKSPACE_PATH)
sys.path.append(WORKSPACE_PATH)

# Pipeline Space
from proj_space import SPACE
SPACE['WORKSPACE_PATH'] = WORKSPACE_PATH
sys.path.append(SPACE['CODE_FN'])

# Available Packages
from recfldtkn.bash_utils import bash_run_recfdltkn_to_hfds

logger = logging.getLogger(__name__)
my_parser = argparse.ArgumentParser(description='Process Input.')
my_parser.add_argument('--cohort_label', type = str, help='Enable step1')
my_parser.add_argument('--record_name', type = str, help='Enable step1')
my_parser.add_argument('--fldtkn_name_list', type = str, help='Enable step1')
my_parser.add_argument('--run_rft_to_hfds', action='store_true', help='Enable step1')
my_parser.add_argument('--run_human_recnum', action='store_true', help='Enable step1')


##################################
if __name__ == "__main__":
    args = my_parser.parse_args()
    cohort_label = args.cohort_label
    #################################
    record_to_FldTknList = {
            'P': ['P-DemoTkn', 
                  'P-Zip3DemoNumeTkn',    'P-Zip3EconNumeTkn', 
                  'P-Zip3HousingNumeTkn', 'P-Zip3SocialNumeTkn', 
                  ],
            
            'PInv': ['PInv-InfoTkn'],

            'Rx': ['Rx-CmpCateTkn', 'Rx-InsCateTkn', 
                   'Rx-ServiceCateTkn', 'Rx-SysCateTkn', 
                   'Rx-QuantN2CTkn', 'Rx-QuantNumeTkn', 
                   'Rx-TherEqCateTkn', 'Rx-DrugBasicCateTkn', 
                   'Rx-PhmBasicCateTkn', 'Rx-PhmZip3HousingNumeTkn', 
                   ],
            
            'EgmClick': [],
            'EgmAuthen': [],
            'EgmCallPharm': [],
            
            'EgmCopay': [],
            'EgmEdu': [],
            'EgmRmd': [],

        }
    #################################

    if args.run_human_recnum:
        '''
        python bash_rft.py --cohort_label 1 --run_human_recnum
        '''
        # TODO


    elif args.run_rft_to_hfds:
        '''
        python bash_rft.py --cohort_label 1 --run_rft_to_hfds
        '''
        for record_name, fldtkn_name_list in record_to_FldTknList.items():
            bash_run_recfdltkn_to_hfds(args, cohort_label, record_name, fldtkn_name_list)

