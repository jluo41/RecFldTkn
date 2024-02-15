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
import datasets
import pandas as pd
from datetime import datetime 

from recfldtkn.ckpd_obs import Ckpd_ObservationS
from recfldtkn.configfn import load_cohort_args
from recfldtkn.obsname import convert_RecObsName_and_CaseTkn_to_CaseObsName
from recfldtkn.loadtools import load_module_variables, update_args_to_list
from recfldtkn.observer import get_RecObsName_to_RecObsInfo, CaseObserverTransformer
from recfldtkn.pipeline_case import pipeline_caseset_to_caseobservation
from recfldtkn.aidstools import get_caseset_to_observe


logger = logging.getLogger(__name__)
recfldtkn_config_path = os.path.join(SPACE['CODE_RFT'], 'config_recfldtkn/')


##################################
my_parser = argparse.ArgumentParser(description='Process Input.')
# Add the arguments

my_parser.add_argument('--case_type',
                    metavar='case_type',
                    default = None, 
                    type=str)

my_parser.add_argument('--group_id',
                    metavar='group_id',
                    default = None, 
                    type=str)

my_parser.add_argument('--record_observations',
                    metavar='record_observations',
                    nargs='+',  
                    default = None, 
                    type=str)

my_parser.add_argument('--case_id_columns',
                    metavar='case_id_columns',
                    nargs='+',  
                    default = None, 
                    type=str)

my_parser.add_argument('--case_tkn',
                    metavar='case_tkn',
                    default = None, 
                    type=str)

my_parser.add_argument('--test',
                    metavar='test',
                    default = None, 
                    type=str)

my_parser.add_argument('--batch_size',
                    metavar='batch_size',
                    default = 1000, 
                    type=str)


if __name__ == '__main__':

    # step 1: get the arguments
    args = my_parser.parse_args()
    CaseTkn = args.case_tkn
    case_type = args.case_type
    group_id = int(args.group_id)
    batch_size = int(args.batch_size)
    case_id_columns = update_args_to_list(args.case_id_columns)
    TEST = True if args.test == 'true' else False
    cohort_args = load_cohort_args(recfldtkn_config_path, SPACE)
    cohort_args['Ckpd_ObservationS'] = Ckpd_ObservationS
    
    # step 2: get the case to be observed
    CaseFolder = os.path.join(SPACE['DATA_TASK'], 'CaseFolder', case_type)
    group_name, ds_case = get_caseset_to_observe(group_id, CaseFolder, case_id_columns, cohort_args)
    print('\n================ group_id and group_name ================')
    print(group_id, group_name)
    print(ds_case)
    
    # step 3: prepare the \Phi's configuration
    print('\n================ Prepare the \Phi  configuration ================')
    Record_Observations_List = update_args_to_list(args.record_observations)
    CaseTkn = args.case_tkn
    CaseObsName = convert_RecObsName_and_CaseTkn_to_CaseObsName(Record_Observations_List, CaseTkn)
    print('Record_Observations_List: ', Record_Observations_List)
    print('CaseTkn: ', CaseTkn) 
    print('CaseObsName: ', CaseObsName)

    record_to_ds_rec = {}
    record_to_ds_rec_info = {}
    use_caseobs_from_disk = True
    case_obs_result = pipeline_caseset_to_caseobservation(ds_case, Record_Observations_List, 
                                                            CaseTkn, 
                                                            SPACE, 
                                                            cohort_args, 
                                                            record_to_ds_rec, 
                                                            record_to_ds_rec_info, 
                                                            use_caseobs_from_disk,
                                                            batch_size)
        
    
    
    RecObsName_to_RecObsInfo, ds_caseobs, fn_caseobs_Phi, CaseTknVocab = case_obs_result
    
    print('\n======== have a look at RecObsName_to_RecObsInfo =========')
    print(RecObsName_to_RecObsInfo)

    # step 4: save the results
    print('\n======== save ds_caseobs and vocabulary =========')
    # this code will only same the new observations made in this turn. 
    fn_caseobs_Phi.save_new_caseobs_to_ds_caseobs()
    # this code will overwrite the old vocabulary. TODO: add the assertion that they should of the same content. 
    pd.DataFrame({CaseObsName: CaseTknVocab}).to_pickle(fn_caseobs_Phi.CaseObsFolder_vocab)
    
    print('\n======== have a look at the random case =========')
    random_int = random.randint(0, len(ds_caseobs))
    print(random_int, ds_caseobs[random_int])
    print(CaseTknVocab)
    print('\n\n')
