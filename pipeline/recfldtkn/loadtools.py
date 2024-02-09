import os
import sys
import pickle
import inspect
import importlib 
import yaml
import numpy as np
import pandas as pd
import datasets
from functools import reduce

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
        ds_rec_info = datasets.concatenate_datasets(linfo)
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
        line = f'{retrieve_name(i)} = {i}'
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



    