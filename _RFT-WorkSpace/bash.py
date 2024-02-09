import subprocess
import multiprocessing
import argparse
import subprocess
import shlex
import os
import sys

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


def generate_run_egm_table_split_command(cohort_label, egm_filename):
    command = f'''
    python run_egm_table_split_drfirst.py \
        --cohort_label {cohort_label} \
        --egm_filename {egm_filename}
    '''
    return command

def generate_run_cohort_info_command(cohort_label):
    '''
    p_i \in P^cohort: basic information for p_i among R_i.
    '''
    command = f'''
    python run_cohort_info_drfirst.py \
        --cohort_label {cohort_label} 
    '''
    return command


def generate_run_rawrec_to_recattr_command(cohort_label, record_name):
    '''
    This is for pipeline A. 
    '''
    command = f'''
    python run_rawrec_to_recattr_drfirst.py \
        --cohort_label {cohort_label} \
        --record_name {record_name}
    '''
    return command


def generate_run_recfldtkn_to_hfds_command(record_name, fldtkn_name_list):
    if len(fldtkn_name_list) == 0:
        command = f'''
        python run_recfldtkn_to_hfds.py \
            --record_name {record_name}
        ''' 
    else: 
        fldtkn_name_list = ' '.join(fldtkn_name_list)
        command = f'''
        python run_recfldtkn_to_hfds.py \
            --record_name {record_name} \
            --fldtkn_name_list {fldtkn_name_list}
        '''
    return command


##################################
my_parser = argparse.ArgumentParser(description='Process Input.')
# Add the arguments
my_parser.add_argument('--step1', action='store_true', help='Enable step1')
my_parser.add_argument('--step2', action='store_true', help='Enable step2')
my_parser.add_argument('--step3', action='store_true', help='Enable step3')
my_parser.add_argument('--step4', action='store_true', help='Enable step4')
my_parser.add_argument('--step5', action='store_true', help='Enable step5')
my_parser.add_argument('--core_num', default=4, type=int)

if __name__ == "__main__":
    args = my_parser.parse_args()
    core_num = args.core_num

    # Step 1: Data Preprocessing 
    if args.step1:
        #################################
        cohort_to_egm_filename_list = [
            [1, "engagement_df_de_identified.csv"],
        ]
        #################################

        command_list = []
        for cohort_to_egm_filename in cohort_to_egm_filename_list:
            cohort_label, egm_filename = cohort_to_egm_filename
            command = generate_run_egm_table_split_command(cohort_label, egm_filename) 
            command_list.append(command)

        # run_multi_process
        run_multi_process(command_list, core_num)

    # Step 2: Cohort Info
    if args.step2:
        #################################
        cohort_label_list = [1]
        #################################

        command_list = []
        for cohort_label in cohort_label_list:
            command = generate_run_cohort_info_command(cohort_label) 
            command_list.append(command)
        # run_multi_process
        run_multi_process(command_list, core_num)


    # Step 3: Pipeline A: from RawRec to RecAttr
    if args.step3:
        #################################
        # pay attention to the orders. 
        # can update this into the record_layers # TODO.
        record_name_list = ['P', 'PInv', 'Rx', 
                            'EgmClick', 'EgmAuthen', 'EgmCallPharm', 
                            'EgmCopay', 'EgmEdu', 'EgmRmd']
        
        cohort_label_list = [1]
        #################################

        for record_name in record_name_list:
            command_list = []
            for cohort_label in cohort_label_list:
                command = generate_run_rawrec_to_recattr_command(cohort_label, record_name)
                command_list.append(command)
            # run_multi_process
            run_multi_process(command_list, core_num)


    # Step 4: for pipeline B: small \phi to get record-level features: z_ik = \phi(r_ik)
    if args.step4:
        #################################
        record_to_FldTknList = {
            'P': ['P-DemoTkn', 
                  'P-Zip3DemoNumeTkn',    'P-Zip3EconNumeTkn', 
                  'P-Zip3HousingNumeTkn', 'P-Zip3SocialNumeTkn', 
                  ],
            
            'PInv': [],

            'Rx': ['Rx-CmpCateTkn', 'Rx-InsCateTkn', 
                   'Rx-QuantN2CTkn', 'Rx-QuantNumeTkn', 
                   'Rx-ServiceCateTkn', 'Rx-SysCateTkn', 
                   ],
            
            'EgmClick': [],
            'EgmAuthen': [],
            'EgmCallPharm': [],
            
            'EgmCopay': [],
            'EgmEdu': [],
            'EgmRmd': [],

        }
        #################################
        command_list = []
        for record_name, fldtkn_name_list in record_to_FldTknList.items():
            command = generate_run_recfldtkn_to_hfds_command(record_name, fldtkn_name_list)
            command_list.append(command)
        
        # run_multi_process
        run_multi_process(command_list, core_num)


