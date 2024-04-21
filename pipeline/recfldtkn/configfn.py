import os
import json
import yaml
import logging
import pandas as pd
# from .ckpd_obs import Ckpd_ObservationS

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s:%(asctime)s:(%(filename)s@%(lineno)d %(name)s)]: %(message)s')


def get_rec_related_size(RecName, cohort_args):
    size = cohort_args['RecName_to_RFT_GROUP_SIZE']['Default']
    RFT_GROUP_SIZE = cohort_args['RecName_to_RFT_GROUP_SIZE'].get(RecName, size)
    
    size = cohort_args['RecName_to_RFT_idx_group_size']['Default']
    idx_group_size = cohort_args['RecName_to_RFT_idx_group_size'].get(RecName, size)
    
    usebucket = cohort_args['RecName_to_RFT_usebucket']['Default']
    usebucket = cohort_args['RecName_to_RFT_usebucket'].get(RecName, usebucket)

    return RFT_GROUP_SIZE, idx_group_size, usebucket


def load_cohort_args(recfldtkn_config_path, SPACE = None, use_inference = False):
    file_path = os.path.join(recfldtkn_config_path, 'Cohort.yaml')
    with open(file_path, 'r') as file: cohort_args = yaml.safe_load(file)
    cohort_args['recfldtkn_config_path'] = recfldtkn_config_path
    
    if SPACE is not None:
        cohort_args['SPACE'] = SPACE
        # cohort_args['rec_folder'] = cohort_args['rec_folder'].replace('$DATA_RFT$', SPACE['DATA_RFT']) 
        # cohort_args['fld_folder'] = cohort_args['rec_folder'].replace('$DATA_RFT$', SPACE['DATA_RFT']) 
        # cohort_args['hfds_folder'] = cohort_args['hfds_folder'].replace('$DATA_RFT$', SPACE['DATA_RFT']) 
        cohort_args['recattr_pyfolder'] = cohort_args['recattr_pyfolder'].replace('$CODE_FN$', SPACE['CODE_FN']) 
        cohort_args['fldtkn_pyfolder'] = cohort_args['fldtkn_pyfolder'].replace('$CODE_FN$', SPACE['CODE_FN']) 
        cohort_args['humanrec_pyfolder'] = cohort_args['humanrec_pyfolder'].replace('$CODE_FN$', SPACE['CODE_FN']) 
        cohort_args['inference_pyfolder'] = cohort_args['inference_pyfolder'].replace('$CODE_FN$', SPACE['CODE_FN']) 
        cohort_args['trigger_pyfolder'] = cohort_args['trigger_pyfolder'].replace('$CODE_FN$', SPACE['CODE_FN']) 
        
        cohort_args['pypath'] = os.path.join(cohort_args['humanrec_pyfolder'], 'humanrec.py')
        for CohortName, CohortConfig in cohort_args['CohortInfo'].items():
            CohortConfig['FolderPath'] = CohortConfig['FolderPath'].replace('$DATA_RAW$', SPACE['DATA_RAW']) 

    if use_inference:
        # CohortInfo = cohort_args['CohortInfo']
        # for shadow_CohortName, CohortConfig in CohortInfo.items(): break 
        # print(shadow_CohortName)
        # CohortConfig = CohortConfig.copy()
        CohortConfig = {}
        CohortName = 'inference'
        cohort_label = 9 # <------- to do
        CohortConfig['cohort_label'] = cohort_label
        CohortConfig['cohort_name'] = CohortName
        CohortConfig['FolderPath'] = os.path.join(SPACE['DATA_RAW'], CohortName)
        cohort_args['CohortInfo'][CohortName] = CohortConfig

    # cohort_args['Ckpd_ObservationS'] = Ckpd_ObservationS
    return cohort_args

def load_record_args(RecName, cohort_args,  use_inference = False, recfldtkn_config_path = None, ):
    SPACE = cohort_args['SPACE']
    recfldtkn_config_path = cohort_args['recfldtkn_config_path']
    file_path = os.path.join(recfldtkn_config_path, 'Record', f'{RecName}.yaml')
    if not os.path.exists(file_path):
        record_args = {}
        with open(file_path, 'w') as file: pass
    else:
        with open(file_path, 'r') as file: record_args = yaml.safe_load(file)
        if record_args is None: record_args = {}
        for Cohort, RecTables_args in record_args.get('CohortInfo', {}).items():
            for TableBase, Table_args in RecTables_args.items():
                Table_args['raw_data_path'] = Table_args['raw_data_path'].replace('$DATA_RAW$', SPACE['DATA_RAW'])
                RecTables_args[TableBase] = Table_args
                
    RFT_GROUP_SIZE, idx_group_size, usebucket = get_rec_related_size(RecName, cohort_args)
    record_args['RFT_GROUP_SIZE'] = RFT_GROUP_SIZE
    record_args['idx_group_size'] = idx_group_size
    record_args['usebucket'] = usebucket
    record_args['GROUP_SIZE'] = RFT_GROUP_SIZE
    # record_args['folder'] = cohort_args['rec_folder']
    # record_args['rec_folder'] = cohort_args['rec_folder']
    record_args['pypath'] = os.path.join(cohort_args['recattr_pyfolder'], f'{RecName}.py')
    record_args['recfldtkn_config_path'] = recfldtkn_config_path
    record_args['yaml_file_path'] = file_path
    
    if 'FldTknInfo' not in record_args:
        record_args['FldTknInfo'] = {} 
        
    if record_args['FldTknInfo'] is None:
        record_args['FldTknInfo'] = {} 

    if use_inference:
        CohortInfo = record_args.get('CohortInfo', {})
        # print('CohortInfo', CohortInfo)
        for shadow_CohortName, shadow_RecTables_args in CohortInfo.items(): break
        
        CohortName = 'inference'
        # print(shadow_CohortName)
        RecTables_args = shadow_RecTables_args.copy()
        for RecTable, Table_args in RecTables_args.items():
            path = Table_args['raw_data_path']
            filename = path.split('/')[-1]  
            new_path = os.path.join(SPACE['DATA_RAW'], 'inference', filename)
            Table_args['raw_data_path'] = new_path
            RecTables_args[RecTable] = Table_args
        # print('RecTables_args', RecTables_args)
        record_args['CohortInfo'][CohortName] = RecTables_args

    return record_args


def load_fldtkn_args(RecName, FldTknName, cohort_args, recfldtkn_config_path = None):
    recfldtkn_config_path = cohort_args['recfldtkn_config_path']
    file_path = os.path.join(recfldtkn_config_path, 'Record', f'{RecName}.yaml')
    logger.info(f'file_path in load_fldtkn_args: {file_path}')
    assert os.path.exists(file_path)
    # with open(file_path, 'r') as file: record_args = yaml.safe_load(file)
    record_args = load_record_args(RecName, cohort_args, recfldtkn_config_path = recfldtkn_config_path)
    
    fldtkn_args = {}
    
    # print(record_args)
    # print([i for i in record_args])
    fldtkn_args = record_args['FldTknInfo'].get(FldTknName, fldtkn_args)
    
    fldtkn_args['SPACE'] = cohort_args['SPACE']
    # RFT_GROUP_SIZE, idx_group_size, usebucket = get_rec_related_size(RecName, cohort_args)
    # fldtkn_args['RFT_GROUP_SIZE'] = RFT_GROUP_SIZE
    # fldtkn_args['idx_group_size'] = idx_group_size
    # fldtkn_args['usebucket'] = usebucket
    # fldtkn_args['GROUP_SIZE'] = RFT_GROUP_SIZE

    AttrFnName_list = FldTknName.split('-')[-1].split('.')
    
    AttrFnName = AttrFnName_list[-1]
    PyFileName = RecName + '_' + AttrFnName
    fldtkn_args['pypath'] = os.path.join(cohort_args['fldtkn_pyfolder'], PyFileName + '.py')
    
    
    AttrFnName_to_Config = {}
    num_proc = 4
    for idx, AttrFnName in enumerate(AttrFnName_list):
        # config = {}
        # PyFileName = RecName + '_' + AttrFnName
        # config['pypath'] = os.path.join(cohort_args['fldtkn_pyfolder'], PyFileName'.py')
        # FldTknNameNew = RecName + '-' + PyFileName # '-'.join(AttrFnName_list[:idx+1])
        # fldtkn_args_prefix = load_fldtkn_args(RecName, FldTknNameNew, cohort_args, recfldtkn_config_path = None)
        fldtkn_args_prefix = record_args['FldTknInfo'][RecName + '-' + AttrFnName]
        fldtkn_args_prefix['pypath'] = os.path.join(cohort_args['fldtkn_pyfolder'], RecName + '_' + AttrFnName + '.py')
        AttrFnName_to_Config[AttrFnName] = fldtkn_args_prefix
        num_proc = fldtkn_args_prefix.get('num_proc', 4)

    # take the last one as the num_proc.
    fldtkn_args['num_proc'] = num_proc
        
    fldtkn_args['AttrFnName_to_Config'] = AttrFnName_to_Config
        
    if len(AttrFnName_list) > 1:
        derv_attr_cols = [RecName + '-' + '.'.join(AttrFnName_list[:idx + 1]) for idx in range(len(AttrFnName_list)-1)]
    else:
        derv_attr_cols = []
    value_cols = list(set(sum([i.get('value_cols', []) for i in AttrFnName_to_Config.values()], [])))
    
    fldtkn_args['derv_attr_cols'] = derv_attr_cols
    fldtkn_args['value_cols'] = value_cols
    
    fldtkn_args['attr_cols'] = record_args['RecIDChain'] + record_args.get('DTCols', []) + value_cols + derv_attr_cols
    fldtkn_args['value_cols'] = value_cols
    fldtkn_args['FldTknName'] = FldTknName
    fldtkn_args['Name'] = FldTknName
    

    if 'external_source_path' in fldtkn_args:
        fldtkn_args['external_source_path'] = fldtkn_args['external_source_path'].replace('$DATA_EXTERNAL$', cohort_args['SPACE']['DATA_EXTERNAL'])
        fldtkn_args['external_source'] = pd.read_pickle(fldtkn_args['external_source_path'])    

    # fldtkn_args['folder'] = cohort_args['fld_folder']
    fldtkn_args['recfldtkn_config_path'] = recfldtkn_config_path
    fldtkn_args['yaml_file_path'] = file_path
    
    
    # FldTknName
    # maybe also add more Demo Examples?
    # for ICL. 
    fldtkn_args['ContextDemo'] = None # TODO, adding the context demo.
    

    return fldtkn_args


def load_rft_config(recfldtkn_config_path, 
                    RecName_list = None, 
                    FldTknName_list = None,
                    SPACE = None, 
                    use_inference = False):
    rft_config = {}
    
    # base yaml
    base_config = load_cohort_args(recfldtkn_config_path, SPACE, use_inference)
    rft_config['base_config'] = base_config

    # record yaml
    rft_config['rec_configs'] = {}
    
    # -- generate the RecName_list
    if RecName_list is None:
        RecName_list = os.listdir(os.path.join(recfldtkn_config_path, 'Record'))
    
    # -- check and get the record yaml and py
    for RecName in RecName_list:
        rec_yaml_path = os.path.join(recfldtkn_config_path, 'Record', f'{RecName}.yaml')
        if not os.path.exists(rec_yaml_path): 
            logger.warning(f'[YAML] {rec_yaml_path} does not exist'); continue
        rec_config = load_record_args(RecName, base_config, 
                                      use_inference = use_inference, 
                                      recfldtkn_config_path = recfldtkn_config_path)
        
        pypath = rec_config['pypath']
        # print(pypath)
        if not os.path.exists(pypath): 
            logger.warning(f'[PY] {pypath} does not exist'); continue
        rft_config['rec_configs'][RecName] = rec_config

    # fldtkn yaml
    rft_config['fldtkn_configs'] = {}

    # -- generate the FldTknName_list
    if FldTknName_list is None:
        FldTknName_list = []
        for RecName in RecName_list:
            rec_config = rft_config['rec_configs'][RecName]
            FldTknName_list += rec_config.get('FldTknInfo', {}).keys()
    
    # -- check and get the fldtkn yaml and py
    for fldtkn in FldTknName_list:
        RecName = fldtkn.split('-')[0]
        fldtkn_config = load_fldtkn_args(RecName, fldtkn, base_config, recfldtkn_config_path)
        pypath = fldtkn_config['pypath']
        # print(RecName)
        # print(pypath)
        if not os.path.exists(pypath): 
            logger.warning(f'[PY] {pypath} does not exist'); continue
        rft_config['fldtkn_configs'][fldtkn] = fldtkn_config
    return rft_config