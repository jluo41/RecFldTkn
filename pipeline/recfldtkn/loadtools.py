import os
import sys
import pickle
import inspect
import importlib 
import yaml
import numpy as np
import pandas as pd
import logging
import datasets
from functools import reduce
import pprint 

logger = logging.getLogger(__name__)

def fetch_trigger_tools(TriggerCaseMethod, SPACE):
    pypath = os.path.join(SPACE['CODE_FN'], 'fn_trigger', f'{TriggerCaseMethod}.py')
    module = load_module_variables(pypath)
    TriggerRecName = module.TriggerRecName
    case_id_columns = module.case_id_columns
    special_columns = module.special_columns
    convert_TriggerEvent_to_Caseset = module.convert_TriggerEvent_to_Caseset
    tools = {}
    tools['TriggerRecName'] = TriggerRecName
    tools['case_id_columns'] = case_id_columns
    tools['special_columns'] = special_columns
    tools['convert_TriggerEvent_to_Caseset'] = convert_TriggerEvent_to_Caseset
    return tools

def fetch_casetag_tools(TagMethod, SPACE):
    pypath = os.path.join(SPACE['CODE_FN'], 'fn_learning', f'{TagMethod}.py')
    module = load_module_variables(pypath)
    InfoRecName = module.InfoRecName
    subgroup_columns = module.subgroup_columns
    fn_case_tagging = module.fn_case_tagging
    tools = {}
    tools['InfoRecName'] = InfoRecName
    tools['subgroup_columns'] = subgroup_columns
    tools['fn_case_tagging'] = fn_case_tagging
    return tools

def fetch_casefilter_tools(FilterMethod, SPACE):
    pypath = os.path.join(SPACE['CODE_FN'], 'fn_learning', f'{FilterMethod}.py')
    module = load_module_variables(pypath)
    fn_case_filtering = module.fn_case_filtering
    tools = {}
    tools['fn_case_filtering'] = fn_case_filtering
    return tools

# def fetch_entry_tools(entry_args, SPACE):
#     tools = {}
#     for entry_name, entry_method in entry_args.items():
#         pypath = os.path.join(SPACE['CODE_FN'], 'fn_learning', f'{entry_method}.py')
#         module = load_module_variables(pypath)
#         # print([i for i in module.MetaDict])
#         tools[entry_name] = module.MetaDict['fn_' + entry_name]
#     return tools


def fetch_entry_tools(entry_method, SPACE):
    pypath = os.path.join(SPACE['CODE_FN'], 'fn_learning', f'{entry_method}.py')
    module = load_module_variables(pypath)
    fn_entry_method_for_finaldata = module.MetaDict['fn_entry_method_for_finaldata']
    return fn_entry_method_for_finaldata


def fetch_fldtkn_phi_tools(RecFldTkn, fldtkn_args, SPACE):

    RecFldTkn = RecFldTkn.replace('-', '_')
    pypath = os.path.join(SPACE['CODE_FN'], 'fn_fldtkn', f'{RecFldTkn}.py')
    module = load_module_variables(pypath)
    
    column_to_top_values = module.column_to_top_values
    item_to_configs = module.item_to_configs
    idx2tkn = module.idx2tkn
    tokenizer_fn = module.tokenizer_fn
    fldtkn_args['item_to_configs'] = item_to_configs
    fldtkn_args['column_to_top_values'] = column_to_top_values
    
    tools = {}
    tools['column_to_top_values'] = column_to_top_values
    tools['item_to_configs'] = item_to_configs
    tools['idx2tkn'] = idx2tkn
    tools['tokenizer_fn'] = tokenizer_fn
    tools['fldtkn_args'] = fldtkn_args  
    return tools


def fetch_caseobs_Phi_tools(name_CasePhi, CaseObsName, SPACE):
    
    pypath = os.path.join(SPACE['CODE_FN'], 'fn_casephi', f'phi_{name_CasePhi}.py')
    module = load_module_variables(pypath)
    get_CO_id = module.get_CO_id
    get_selected_columns = module.get_selected_columns
    get_CO_vocab = module.get_CO_vocab
    fn_CasePhi = module.fn_CasePhi
    CO_Folder = os.path.join(SPACE['DATA_CaseObs'], CaseObsName)

    tools = {}
    tools['get_CO_id'] = get_CO_id
    tools['get_selected_columns'] = get_selected_columns
    tools['get_CO_vocab'] = get_CO_vocab
    tools['fn_CasePhi'] = fn_CasePhi
    tools['CO_Folder'] = CO_Folder
    
    return tools 

def fetch_casefeat_Gamma_tools(CaseTaskOp, CaseFeatName, SPACE):

    pypath = os.path.join(SPACE['CODE_FN'], 'fn_casegamma',  f'gamma_{CaseTaskOp}.py')
    module = load_module_variables(pypath)
    get_CF_id = module.get_CF_id
    get_CF_vocab = module.get_CF_vocab
    fn_CaseGamma = module.fn_CaseGamma
    CF_Folder = os.path.join(SPACE['DATA_CaseFeat'], CaseFeatName)

    tools = {}
    tools['get_CF_id'] = get_CF_id
    tools['get_CF_vocab'] = get_CF_vocab
    tools['fn_CaseGamma'] = fn_CaseGamma
    tools['CF_Folder'] = CF_Folder
    return tools

def fetch_TriggerEvent_tools(TriggerCaseMethod, SPACE):
    pypath = os.path.join(SPACE['CODE_FN'], 'fn_trigger', f'{TriggerCaseMethod}.py')
    
    module = load_module_variables(pypath)
    tools = {}
    tools['TriggerRecName'] = module.TriggerRecName
    tools['case_id_columns'] = module.case_id_columns
    tools['special_columns'] = module.special_columns
    tools['convert_TriggerEvent_to_Caseset'] = module.convert_TriggerEvent_to_Caseset
    return tools


def filter_with_cohort_label(df, cohort_label, cohort_args):
    RootID = cohort_args['RootID']
    filter_fn = lambda x: str(x)[:-cohort_args['RootIDLength']] == str(cohort_label)
    df = df[df[RootID].astype(str).apply(filter_fn)].reset_index(drop = True)
    return df


def load_ds_rec_and_info(record_name, cohort_args, cohort_label_list = None):
    SPACE = cohort_args['SPACE']
    cohort_list = [i for i in os.listdir(SPACE['DATA_RFT'])]
    if cohort_label_list is not None:
        cohort_label_list = [str(i) for i in cohort_label_list]
        cohort_list = [i for i in cohort_list if i.split('-')[0] in cohort_label_list]
    l = []
    linfo = []
    for cohort_full_name in cohort_list:
        data_folder = os.path.join(SPACE['DATA_RFT'], cohort_full_name, record_name + '_data')
        # logger.info(f'Load from disk: {data_folder} ...')
        if not os.path.exists(data_folder):
            logger.info(f'No such folder: {data_folder} ...')
            continue
        ds_rec = datasets.Dataset.load_from_disk(data_folder)
        l.append(ds_rec)
        info_folder = os.path.join(SPACE['DATA_RFT'], cohort_full_name, record_name + '_info')
        if os.path.exists(info_folder):
            ds_rec_info = datasets.Dataset.load_from_disk(info_folder)
            linfo.append(ds_rec_info)
    ds_rec = datasets.concatenate_datasets(l)
    if len(linfo) == 0:
        ds_rec_info = None
    else:
        ########################## BIGGEST BUG EVER ############################
        # ds_rec_info = datasets.concatenate_datasets(linfo)
        ########################################################################

        assert len(linfo) == len(l)
        num_rec = 0
        num_ptt = 0
        linfo_new = []
        for idx, ds_rec_info in enumerate(linfo):
            
            df_rec_info = ds_rec_info.to_pandas()   
            df_rec_info['PID_idx'] = num_ptt + df_rec_info['PID_idx']
            df_rec_info['interval'] = df_rec_info['interval'].apply(lambda x: [i + num_rec for i in x])
            ds_rec_info = datasets.Dataset.from_pandas(df_rec_info)
            linfo_new.append(ds_rec_info)

            num_rec = num_rec + len(l[idx])
            num_ptt = num_ptt + len(ds_rec_info)
        
        ds_rec_info = datasets.concatenate_datasets(linfo_new)

    return ds_rec, ds_rec_info


def update_args_to_list(x):
    if x is None:
        return []
    elif type(x) == list:
        return x    
    elif type(x) == str:
        return [x]
    else:
        raise ValueError(f'Unknown type: {type(x)}')
    

def retrieve_name(var):
    """
    Gets the name of var. Does it from the out most frame inner-wards.
    :param var: variable to get name from.
    :return: string
    """
    for fi in reversed(inspect.stack()):
        names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var and var_name[0] != '_']
        # print(names)
        if len(names) > 0:
            return names[0]
        

def convert_variables_to_pystirng(string_variables = [], 
                                  iterative_variables = [], 
                                  fn_variables = [], 
                                  prefix = ['import pandas as pd', 'import numpy as np']):
    L = prefix
    
    for i in string_variables:
        line = f'{retrieve_name(i)} = "{i}"'
        L.append(line)
        
    for i in iterative_variables:
        if type(i) == dict:
            pretty_str = pprint.pformat(i, 
                                indent=4, 
                                width=100, 
                                sort_dicts=False, 
                                compact=True,)
        else:
            pretty_str = str(i)
        # i = pretty_str
        line = f'{retrieve_name(i)} = {pretty_str}'
        L.append(line)
        
    for i in fn_variables:
        line = f'{i.fn_string}'
        L.append(line)
        
    D_str = "\n\nMetaDict = {\n" + ',\n'.join(
                    ['\t' + f'"{retrieve_name(i)}": {retrieve_name(i)}'
                 for i in string_variables + iterative_variables + fn_variables]
                ) + "\n}"
     
    python_strings = '\n\n'.join(L) + D_str
    
    return python_strings


def load_module_variables(file_path):

    # Extract the directory from the file path
    module_dir = os.path.dirname(file_path)

    # Extract the module name from the file path
    module_name = os.path.basename(file_path).split('.')[0]

    # Add the module directory to sys.path
    if module_dir not in sys.path:
        sys.path.append(module_dir)

    # Import the module using its name
    module = importlib.import_module(module_name)

    return module


def add_key_return_dict(dictionary, key, value):
    dictionary = dictionary.copy()
    dictionary[key] = value
    return dictionary

def find_timelist_index(dates, DT):
    low, high = 0, len(dates) - 1
    if DT < dates[0]:
        # return -1  # DT is smaller than the first date in the list
        return 0     # DT is smaller than the first date in the list
    if DT > dates[-1]:
        # return len(dates)  # DT is larger or equal to the last date in the list
        return len(dates)    # DT is larger or equal to the last date in the list

    while low <= high:
        mid = (low + high) // 2
        if dates[mid] < DT:
            low = mid + 1
        else:
            high = mid - 1
    return low

# ========================================== deprecate ==========================================

def update_yaml(file_path, key, value):
    with open(file_path, 'r') as file:
        content = yaml.safe_load(file)
    content[key] = value
    with open(file_path, 'w') as file:
        yaml.dump(content, file)


def save_to_pickle(obj, filename):
    with open(filename, 'wb') as f:  # write in binary mode
        pickle.dump(obj, f)


def load_from_pickle(filename):
    with open(filename, 'rb') as f:  # read in binary mode
        return pickle.load(f)
    

def get_df_bucket_from_settings(bucket_file, RecChain_ARGS, RecInfo_ARGS):
    # 1 ----------------- get df_prefix
    L = []
    # RecNameID_Chain = [f'{i}ID' for i in RecName_Chain]
    RecNameID_Chain = [i for i in RecChain_ARGS]
    for idx, RID in enumerate(RecChain_ARGS):
        RecInfo = RecChain_ARGS[RID]
        # folder, Name = RecName_Chain_To_RecFolderName.get(RecName, (rec_folder, RecName))
        folder, RecName = RecInfo['folder'], RecInfo['RecName']
        df = pd.read_pickle(os.path.join(folder, RecName, bucket_file))
        df = df[RecNameID_Chain[:idx+ 1]].astype(str).drop_duplicates()
        L.append(df)
        
    df_prefix = reduce(lambda left, right: pd.merge(left, right, how = 'left'), L)
    # print(df_prefix)
    
    # fill the missing Rec with missing ID.
    for RID in RecNameID_Chain:
        s = 'M' + pd.Series(df_prefix.index).astype(str)
        df_prefix[RID] = df_prefix[RID].fillna(s)

    
    # 2 ----------------- get df_data
    # RecLevel = RecNameID_Chain[-1][:-2]
    # RecLevelID = f'{RecLevel}ID'
    RecLevelID = RecNameID_Chain[-1]

    L = []
    # for RecName, FldList in RecTableName2FldColumns_Dict.items():
    for idx, RecElmt in enumerate(RecInfo_ARGS):
        RecElmt_ARGS = RecInfo_ARGS[RecElmt]
        folder = RecElmt_ARGS['folder']
        RecName = RecElmt_ARGS['RecName']
        FldList = RecElmt_ARGS['Columns']
        
        # read df
        path = os.path.join(folder, RecName, bucket_file)
        if not os.path.exists(path): print(f'empty path: {path}'); continue
        df = pd.read_pickle(path)
        
        # select columns
        if FldList == 'ALL': FldList = list(df.columns)
        full_cols = [i for i in RecNameID_Chain if i not in FldList] + FldList
        full_cols = [i for i in full_cols if i in df.columns]
        df = df[full_cols].reset_index(drop = True)
        
        # update columns
        for RecID in RecNameID_Chain:
            if RecID in df.columns: df[RecID] = df[RecID].astype(str)

        # downstream df_data with df_prefix if RecLevelID is not in ID
        # eg: RecLevelID is Encounter-level, but df is Patient-level.
        if RecLevelID not in df.columns:
            on_cols = [i for i in df_prefix.columns if i in df.columns]
            df = pd.merge(df_prefix, df, on = on_cols, how = 'left')

        # change df to RecLevelID for each row.
        df = pd.DataFrame([{RecLevelID: RecLevelIDValue, RecElmt: df_input} 
                           for RecLevelIDValue, df_input in df.groupby(RecLevelID)])
        
        L.append(df)
        
    # uncomment this is you want to check
    # for df in L: print(df.shape)
    
    # 3 ----------------- get df_data
    # Merge the dataframes in the list using reduce, here we should use OUTER!
    # <!!!!!!!!!!!!!!!! here is the data only, so we use outer !!!!!!!!!!!!!!!!>
    df_data = reduce(lambda left, right: pd.merge(left, right, on=RecLevelID, how = 'outer'), L)

    # 4 ----------------- get df_whole
    # <!!!!!!!!!!!!!!!! here we only focus rows in df_prefix, so we use left !!!!!!!!!!!!!!!!>
    df_whole = pd.merge(df_prefix, df_data, on = RecLevelID, how = 'left')
    
    return df_whole


def get_compressed_df(df_rec, full_recfldgrn_name, prefix_ids):
    '''
        df_rec: with columns: prefix_ids + focal_ids + full_recfldgrn_name
            prefix_ids: [PID, ECID] 
            focal_ids: [PNID], current primary key
            full_recfldgrn_name: `PrimaryNote - Section - SectSent@Sentence - TkGrn`

            prefix_ids_new: [PID] 
            focal_ids_new: [ECID], current primary key
            full_recfldgrn_name_new: `EC - PrimaryNote - Section - SectSent@Sentence - TkGrn`
    '''
    CompressParent_ID_list = prefix_ids # parent layer's ID which will be used to compress. 
    CompressParent_ID = prefix_ids[-1]
    prefix_ids_new = prefix_ids[:-1]   # the new prefix_ids. 

    df_rec_new = pd.DataFrame(df_rec.groupby(CompressParent_ID_list).apply(lambda x: x.to_dict('list')).to_list())

    for col in prefix_ids: 
        df_rec_new[col] = df_rec_new[col].apply(lambda x: x[0])
        
    full_recfldgrn_name_new = CompressParent_ID.replace('ID', '') + '-' + full_recfldgrn_name
    df_rec_new = df_rec_new.rename(columns = {full_recfldgrn_name: full_recfldgrn_name_new})
    df_rec_new = df_rec_new[prefix_ids + [full_recfldgrn_name_new]]
    return df_rec_new, full_recfldgrn_name_new, prefix_ids_new


def get_highorder_input_idx(df, recfldgrn_sfx, prefix_ids, focal_ids):
    '''
        df: raw data with column: prefix_ids + focal_ids + [recfldgrn_sfx]
        recfldgrn_sfx: xxx_wgt, or xxx_tknidx, or xxx_fldidx
        prefix_ids: [PID, ECID, PNID, PNSectID]
        focal_ids: [SectSentID]
    '''
    # recfldgrn = recfldgrn_sfx.split('_')[0]
    # rec, field = recfldgrn.split('-')[0].split('@')
    # grain = recfldgrn.split('-')[-1]
    # top_id = df.columns[0] # need to double check this.
    if len(prefix_ids) > 0:
        prefix_ids_c = prefix_ids.copy() # _c: copy (meaningless)
        df_Rec = df[prefix_ids_c + [recfldgrn_sfx]]
        for i in range(len(prefix_ids_c)):
            df_Rec, recfldgrn_sfx, prefix_ids_c = get_compressed_df(df_Rec, recfldgrn_sfx, prefix_ids_c)
    else:
        df_Rec = df[focal_ids + [recfldgrn_sfx]]

    df_p = df_Rec.reset_index(drop = True) # p: patient.
    return df_p



    