import subprocess
import multiprocessing
import argparse
import subprocess
import shlex
import os
import sys
from pprint import pprint 
from .configfn import load_cohort_args
from .obsname import convert_RecObsName_and_CaseTkn_to_CaseObsName
from .loadtools import load_module_variables, update_args_to_list
from .constaidatatools import convert_case_observations_to_co_to_observation

def run_command_fn(command):
    try:
        # Split command for Windows compatibility
        if os.name == 'nt':  # Windows
            args = shlex.split(command)
        else:  # Unix/Linux
            args = command

        try:
            subprocess.run(args, shell=True, check=True)
        except KeyboardInterrupt:
            # Handle KeyboardInterrupt in worker process if needed
            print("Worker process interrupted")
            return None
    except subprocess.CalledProcessError as e:
        print('\n\n============= Error =============\n')
        print(e)
        print(command)
        print('\n=================================\n\n')

def run_multi_process(command_list, core_num):  
    try:
        pool = multiprocessing.Pool(processes=core_num)
        result = pool.map(run_command_fn, command_list)  
        pool.close()  # No more tasks will be submitted to the pool
        pool.join()   # Wait for the worker processes to exit
    except KeyboardInterrupt:
        print("Interrupted by user, terminating the pool")
        pool.terminate()
        pool.join()
        result = None
    return result  



def bash_run_case_observation(args, 
                            case_type = None, 
                            case_observations = None, 
                            case_id_columns = None, 
                            batch_size = 1000,
                            GROUP_NUM = 10):
    
    # 1. group
    group_id_list = args.group_id_list
    core_num = args.core_num
    if 'all' in group_id_list: group_id_list = [i for i in range(GROUP_NUM)]
    print('group_id_list ---->', group_id_list)

    # 2. case_type
    if case_type is None: case_type = args.case_type
    print('case_type ---->', case_type)

    # 3. case_id_columns
    if case_id_columns is None: case_id_columns = update_args_to_list(args.case_id_columns)
    print('case_id_columns ---->', case_id_columns)

    # 4. case_observations 
    if case_observations is None: case_observations = update_args_to_list(args.case_observations)
    _, co_to_CaseObsNameInfo = convert_case_observations_to_co_to_observation(case_observations)
    print('case_observations ---->', case_observations)

    command_list = []
    for group_id in group_id_list:
        # for record_observations in record_observations_list:
        for co, CaseObsNameInfo in co_to_CaseObsNameInfo.items():
            record_observations = CaseObsNameInfo['Record_Observations_List']
            case_tkn = CaseObsNameInfo['CaseTkn']
            record_observations_list = ' '.join(record_observations)
            case_id_columns_list = ' '.join(case_id_columns)

            command = f'''
            python ../run_case_observations.py \
                --group_id {group_id} \
                --case_type {case_type} \
                --case_id_columns {case_id_columns_list} \
                --record_observations {record_observations_list} \
                --case_tkn {case_tkn} \
                --batch_size {batch_size}
            '''

            print(command)
            command_list.append(command)
            if len(command_list) == core_num:
                # pprint(command_list)
                run_multi_process(command_list, core_num)
                command_list = []
            
    # pprint(command_list)
    run_multi_process(command_list, core_num)



def bash_run_case_taskop(args, 
                        case_type_list = None, 
                        case_observations = None, 
                        case_id_columns = None, 
                        case_taskop = None, 
                        post_process = None, 
                        batch_size = 1000,
                        GROUP_NUM = 10, 
                        ):
    # 1. group
    group_id_list = args.group_id_list
    core_num = args.core_num
    if 'all' in group_id_list: 
        group_id_list = [i for i in range(GROUP_NUM)]
    print('group_id_list ---->', group_id_list)
    # 2. case_type
    if case_type_list is None: case_type_list = args.case_type_list
    print('case_type_list ---->', case_type_list)
    # 3. case_id_columns
    if case_id_columns is None: case_id_columns = update_args_to_list(args.case_id_columns)
    print('case_id_columns ---->', case_id_columns)
    # 4. case_observations 
    if case_observations is None: case_observations = update_args_to_list(args.case_observations)
    # 5. case_taskop 
    if case_taskop is None: case_taskop = args.case_taskop
    # 6. case_taskop 
    if case_taskop is None: case_taskop = args.case_taskop
    # 7. post_process
    if post_process is None: post_process = args.post_process

    command_list = []
    for group_id in group_id_list:
        group_id_list_for_command = [group_id]
        
        group_id_list = group_id_list_for_command
        group_id_list = ' '.join([str(i) for i in group_id_list])
        case_type_list_incmd = ' '.join(case_type_list)
        case_id_columns_list = ' '.join(case_id_columns)   
        case_observations_list = ' '.join(case_observations)
        # in the workspace folder.
        command = f'''
        python ../run_case_taskop.py \
            --group_id_list {group_id_list} \
            --case_type_list {case_type_list_incmd} \
            --case_id_columns {case_id_columns_list} \
            --case_observations {case_observations_list} \
            --case_taskop {case_taskop} \
            --post_process {post_process} \
            --batch_size {batch_size}
        '''
        print(command)
        command_list.append(command)
        # if len(command_list) == core_num:
        #     # pprint(command_list)
        #     run_multi_process(command_list, core_num)
        #     command_list = []

    # pprint(command_list)
    assert core_num == 1
    run_multi_process(command_list, core_num) 

