import os
import logging
import datasets
import shutil
import pandas as pd
from functools import reduce
from .configfn import load_cohort_args, load_record_args, load_fldtkn_args
from .loadtools import load_module_variables, load_ds_rec_and_info 
from .observer import Tokenizer_Transform, DS_Rec_Info_Generator
# Setup basic configuration for logging

logger = logging.getLogger(__name__)


def get_RawName_to_dfRawPath(OneCohort_config, rft_config):
    base_config = rft_config['base_config']

    # Section: Load the cohort information
    ###############################
    pypath = base_config['pypath']
    module = load_module_variables(pypath)
    selected_source_file_suffix_list = module.selected_source_file_suffix_list
    get_tablename_from_raw_filename = module.get_tablename_from_raw_filename
    ###############################

    FolderPath = OneCohort_config['FolderPath']
    assert os.path.exists(FolderPath)

    # file_list = [i for i in os.listdir(FolderPath) if i.split('.')[-1] in selected_source_file_suffix_list]
    file_list = [i for i in os.listdir(FolderPath) if any([sfx in i for sfx in selected_source_file_suffix_list])]
        
    fullfile_list = [os.path.join(FolderPath, i) for i in file_list]
    logger.info(f'{FolderPath} <-- FolderPath')
    logger.info(f'{len(fullfile_list)} <--- fullfile_list')

    RawName_to_dfRawPath = {}
    for file_path in fullfile_list:
        RawName = get_tablename_from_raw_filename(file_path)
        RawName_to_dfRawPath[RawName] = file_path
    return RawName_to_dfRawPath


def read_column_value_counts_by_chunk(RawRootID, 
                                      chunk_size, 
                                      file_path, 
                                      get_rawrootid_from_raw_table_column, 
                                      get_tablename_from_raw_filename,
                                      RawName = None, 
                                      dfRaw = None, 
                                      ):
    if type(dfRaw) != pd.DataFrame:
        columns = pd.read_csv(file_path, nrows=0).columns
    else:
        columns = dfRaw.columns 
    
    id_column = get_rawrootid_from_raw_table_column(columns)
    logger.info(f'id_column: {id_column}')

    if id_column is None:
        results = pd.DataFrame(columns = [RawRootID, 'RawName', 'RecNum'])
        return results 
    
    if type(dfRaw) == pd.DataFrame:
        result = dfRaw[id_column].value_counts()
    else:
        li = [chunk[id_column].value_counts() 
              for chunk in pd.read_csv(file_path, usecols = [id_column], chunksize=chunk_size, low_memory=False)]
        result = pd.concat(li)
        result = result.groupby(result.index).sum()

    if RawName is None:
        RawName = get_tablename_from_raw_filename(file_path)
    result = result.reset_index().rename(columns = {'count': 'RecNum', id_column: RawRootID})
    result['RawName'] = RawName
    return result


def get_CohortLevel_df_Human2RawRecNum(OneCohort_config, 
                                       rft_config, 
                                       RawName_to_dfRaw):

    ################## CohortInfo ##################
    cohort_label = OneCohort_config['cohort_label']
    cohort_name = OneCohort_config['cohort_name']
    logger.info(f'=======cohort_label-{cohort_label}: cohort_name-{cohort_name}=======')
    logger.info(OneCohort_config)
    

    # Section: Load the cohort information
    ###############################
    base_config = rft_config['base_config']
    RawRootID = base_config['RawRootID']
    RootID = base_config['RootID']  
    RootIDLength = base_config['RootIDLength']
    chunk_size = 100000
    pypath = base_config['pypath']
    module = load_module_variables(pypath)
    excluded_raw_table_names = module.excluded_raw_table_names
    get_tablename_from_raw_filename = module.get_tablename_from_raw_filename
    get_rawrootid_from_raw_table_column = module.get_rawrootid_from_raw_table_column
    ###############################

    Result_List = []
    for RawName, info in RawName_to_dfRaw.items():
        if info is None: 
            logger.info(f"'{RawName}' # is None"); continue

        is_larger_than_1GB = False
        if type(info) == str:
            assert os.path.exists(info)
            file_path = info
            
            # check whether the file is empty
            if os.stat(file_path).st_size == 0: 
                logger.info(f"'{file_path}' # emtpy file"); continue
            
            # handle different file types
            if file_path.split('.')[-1] == 'csv': 
                file_size = os.path.getsize(file_path)
                is_larger_than_1GB = file_size > 1_073_741_824
                if not is_larger_than_1GB:
                    dfRaw = pd.read_csv(file_path)
                else:
                    dfRaw = None
                    logger.info(f"'{file_path}' # larger than 1GB")

            elif '.csv.gz' in file_path:
                dfRaw = None
                logger.info(f"'{file_path}' # use csv.gz file")

            elif file_path.split('.')[-1] == 'parquet':
                dfRaw = pd.read_parquet(file_path)
            elif file_path.split('.')[-1] == 'p':
                dfRaw = pd.read_pickle(file_path)
            else:
                raise ValueError(f'file type not supported: {file_path}')
            
        elif type(info) == pd.DataFrame:
            dfRaw = info
            file_path = None

        else:
            raise ValueError(f'file type not supported: {type(info)}')

        # pass if the dfRaw is empty
        if type(dfRaw) == pd.DataFrame:
            if len(dfRaw) == 0: continue 

        # have a double check for the large file.
        if is_larger_than_1GB is True:
            assert file_path.split('.')[-1] == 'csv'
            logger.info(f"processing large file: '{file_path}'")
        
        result = read_column_value_counts_by_chunk(RawRootID, chunk_size, file_path, 
                                                   get_rawrootid_from_raw_table_column, 
                                                   get_tablename_from_raw_filename,
                                                   RawName, 
                                                   dfRaw)
        
        logger.info(f"{RawName}:'path-{file_path}' # {result.shape}")
        Result_List.append(result)

    logger.info(f'{len(Result_List)} <---- types of dfRec so far')
    
    # Section: Concatenate the results each row is a patient.
    df_all = pd.concat(Result_List, ignore_index=True)
    df_pivot = df_all.pivot(index=RawRootID, columns='RawName', values='RecNum').reset_index()
    
    # Section: Filtering the df_Human who does not have any records. 
    recname_cols = [i for i in df_pivot.columns if i != RawRootID]
    # -- only consider the records that is in the included_cols. 
    included_cols = [i for i in recname_cols if i not in excluded_raw_table_names]
    rec_count = df_pivot[included_cols].sum(axis = 1)
    df_Human = df_pivot[rec_count > 0].reset_index(drop = True)
    df_Human['TotalRecNum'] = df_Human[included_cols].sum(axis = 1)
    logger.info(f"The patient cohort size: {len(df_Human)}")

    # Section: Adding and Updating the RootID. 
    CohortLabel = OneCohort_config['cohort_label']
    df_Human[RootID] = range(1, len(df_Human) + 1)
    df_Human[RootID] = df_Human[RootID].apply(lambda x: int(str(CohortLabel) + str(x).zfill(RootIDLength)))
    df_Human['CohortLabel'] = CohortLabel
    cols = ['PID'] + [i for i in df_Human.columns if i not in ['PID']]
    df_Human = df_Human[cols].reset_index(drop = True)

    return df_Human


def get_parentRecord_info(rec_config, rft_config, df_Human, RecName_to_dsRec = {}):

    base_config = rft_config['base_config']

    RootID = base_config['RootID']
    RawRootID = base_config['RawRootID']

    if rec_config['ParentRecName'] is not None:
        # get the parent record information
        prt_record_args = load_record_args(rec_config['ParentRecName'], base_config)

        if rec_config['ParentRecName'] in RecName_to_dsRec:
            ds_Prt = RecName_to_dsRec[rec_config['ParentRecName']]
        else:
            ds_Prt, _ = load_ds_rec_and_info(rec_config['ParentRecName'], base_config)

        columns = prt_record_args['RecIDChain']
        columns = columns + [i for i in prt_record_args['RawRecID'] if i not in columns]
        df_Prt = ds_Prt.select_columns(columns).to_pandas()

        # update df_Human
        df_Human = df_Human[df_Human[RootID].isin(df_Prt[RootID].to_list())].reset_index(drop = True)

    else:
        # print(' no parent record')
        # get the parent record information
        prt_record_args = {}
        prt_record_args['RawRecID'] = base_config['RawRootID']
        df_Prt = df_Human[[RootID, RawRootID]]
        df_Human = df_Human

    parent_results = {
        'prt_record_args': prt_record_args,
        'df_Prt': df_Prt,
        'df_Human': df_Human
    }
    return parent_results

def get_df_HumanSelected_from_OneCohortRecArgs(rec_config, 
                                               RawName_to_RawConfig, 
                                               OneCohort_config, 
                                               df_Human,
                                               base_config):
    RootID = base_config['RootID']
    RawRootID = base_config['RawRootID']
    cohort_label = OneCohort_config['cohort_label']
    idx_group_size = rec_config['idx_group_size']
    RawName_List = [RawName for RawName in RawName_to_RawConfig]
    dfx = df_Human[[RootID, RawRootID, 'CohortLabel'] + RawName_List]
    dfx = dfx[dfx['CohortLabel'] == cohort_label].reset_index(drop = True)
    dfx = dfx[(dfx[RawName_List] > 0).mean(axis = 1) == 1].reset_index(drop = True)
    dfx['index_group'] = (dfx.index / idx_group_size).astype(int)
    df_HumanSelected = dfx
    return df_HumanSelected


def load_HumanRawTable(RawRootID_to_RawNameRecNum, 
                       RawName, 
                       RawRootID, 
                       raw_data_path, 
                       df_rawrec,
                       raw_columns, 
                       chunk_size):

    PatientID_list = [i for i in RawRootID_to_RawNameRecNum]
    total_RecNum = sum([v for k, v in RawRootID_to_RawNameRecNum.items()])

    if type(df_rawrec) == pd.DataFrame:
        df_HumanRec = df_rawrec
        df_HumanRec = df_HumanRec[df_HumanRec[RawRootID].isin(PatientID_list)].reset_index(drop = True)
    else:
        df_HumanRec = pd.DataFrame()
        for chunk in pd.read_csv(raw_data_path, chunksize=chunk_size, low_memory=False):
            chunk = chunk[chunk[RawRootID].isin(PatientID_list)]
            df_HumanRec = pd.concat([df_HumanRec, chunk])
            if len(df_HumanRec) == total_RecNum:
                break
            if len(df_HumanRec) > total_RecNum:
                raise ValueError(f'{RawName} more than given number {total_RecNum}')
        df_HumanRec = df_HumanRec.reset_index(drop = True)

    if type(raw_columns) == list: 
        for col in raw_columns:
            if col not in df_HumanRec.columns:
                logger.warning(f'{col} not in the raw data')
        df_HumanRec = df_HumanRec[raw_columns].reset_index(drop = True)

    return df_HumanRec


def get_HumanRawRec_for_HumanGroup(df_HumanGroup, RawName_to_RawConfig, RawName_to_dfRaw, base_config):
    L = []
    for RawName, RawConfig in RawName_to_RawConfig.items():
        RawName = RawConfig['RawName']
        raw_columns = RawConfig['raw_columns']
        chunk_size = RawConfig.get('chunk_size', 100000)

        # print(RawName, RawConfig)
        # print(df_HumanGroup.head())
        RawRootID_to_RawNameRecNum = dict(zip(df_HumanGroup[base_config['RawRootID']], df_HumanGroup[RawName]))
        
        InfoRaw = RawName_to_dfRaw[RawName]
        if type(InfoRaw) == str:
            raw_data_path = InfoRaw
            assert os.path.exists(raw_data_path)
            file_size = os.path.getsize(raw_data_path)
            assert file_size > 0
            logger.info(f'RawName "{RawName}" from file: {raw_data_path}')
            df_rawrec = None 
        else:
            df_rawrec = InfoRaw
            raw_data_path = None

        df_HumanRawRec_of_RawTable = load_HumanRawTable(RawRootID_to_RawNameRecNum, 
                                                        RawName, 
                                                        RawConfig['RawRootID'], 
                                                        raw_data_path, 
                                                        df_rawrec,
                                                        raw_columns, 
                                                        chunk_size)
        
        
        if RawConfig['RawRootID'] != base_config['RawRootID']:
            base_RawRootID = base_config['RawRootID']   
            rawname_RawRootID = RawConfig['RawRootID']  
            logger.info(f'RawRootID is different from "Base" and "RecName": {rawname_RawRootID} != {base_RawRootID}')
        df = df_HumanRawRec_of_RawTable.rename(columns = {RawConfig['RawRootID']: base_config['RawRootID']})
        L.append(df)
    df_HumanRawRec = reduce(lambda left, right: pd.merge(left, right, on=base_config['RawRootID'], how='outer'), L)
    return df_HumanRawRec


def post_record_process(df, record_args):
    # ---------------------------
    # x. merge with the parent record (a must except Human Records)
    # print(df.shape)
    df_Prt = record_args['df_Prt']
    # print(df_Prt.shape)
    prt_record_args = record_args['prt_record_args']
    df_merged = pd.merge(df_Prt, df, how = 'inner', on = prt_record_args['RawRecID'])
    # print(df_merged.shape)
    df = df_merged

    # y. sort the table by Parent IDs and DT
    RecDT = record_args.get('RecDT', None)
    RecID_Chain = prt_record_args.get('RecIDChain', [])
    if RecDT is not None:
        sorted_cols = RecID_Chain + [RecDT]
    else: 
        sorted_cols = RecID_Chain
    if len(sorted_cols) > 0:
        df = df.sort_values(sorted_cols).reset_index(drop = True)

    # z. create a new column for RecID
    RecID = record_args['RecID']
    PrtRecID = prt_record_args.get('RecID', None)
    if RecID not in df.columns:
        assert PrtRecID is not None
        df[RecID] = df[PrtRecID].astype(str) + '-' + df.groupby(PrtRecID).cumcount().astype(str)
    #-------------------
    return df


def pipeline_for_RecName(rec_config,
                         OneCohort_config,  
                         rft_config,
                         df_Human, 
                         RawName_to_dfRaw,
                         RecName_to_dsRec = {}):

    # 1. basic information
    # cohort information
    cohort_label = OneCohort_config['cohort_label']
    cohort_name = OneCohort_config['cohort_name']
    

    # base information
    base_config = rft_config['base_config']
    RootID = base_config['RootID']
    RawRootID = base_config['RawRootID']
    RootIDLength = base_config['RootIDLength']
    RecName = rec_config['RecName']

    # df_Human information
    df_Human = df_Human.copy()

    # 2. get the pipeline basic function
    pypath = rec_config['pypath']
    assert os.path.exists(pypath)
    logger.info(f'load recattr pipeline from: {pypath} ...')
    module = load_module_variables(pypath)
    RawRec_to_RecAttr_fn = module.RawRec_to_RecAttr_fn
    attr_cols = rec_config['attr_cols']
    assert len(attr_cols) > 0
    logger.info(f'attr_cols: ...')
    logger.info(attr_cols)

    # 3. check the existence of parent record.
    parentResult = get_parentRecord_info(rec_config, rft_config, df_Human, RecName_to_dsRec)
    prt_record_args = parentResult['prt_record_args']
    df_Prt = parentResult['df_Prt']
    df_Human = parentResult['df_Human']

    # and add the df_Prt to the record_args.
    logger.info(f'df_Prt shape: {df_Prt.shape}')
    df_Prt = df_Prt[df_Prt[RootID].astype(str).apply(lambda x: str(x)[:-RootIDLength] == str(cohort_label))].reset_index(drop = True)
    rec_config['df_Prt'] = df_Prt
    rec_config['prt_record_args'] = prt_record_args
    logger.info(f'df_Prt shape: {df_Prt.shape}')

    # and update df_Human based on df_Prt. 
    df_Human = df_Human[df_Human[RootID].isin(df_Prt[RootID].to_list())].reset_index(drop = True)
    
    # 5. get RawName_to_RawConfig and check the source data existence
    RawInfo = rec_config['RawInfo']
    RawName_to_RawConfig = {} # for a record, we need a dictionary RawName_to_RawConfig
    for RawTable, RawConfig in RawInfo.items():
        RawName = RawConfig['RawName']
        assert RawName in RawName_to_dfRaw
        InfoRaw = RawName_to_dfRaw[RawName]
        if type(InfoRaw) == str: 
            file_path = InfoRaw
            assert os.path.exists(file_path)
            file_size = os.path.getsize(file_path)
            assert file_size > 0
            logger.info(f'RawName "{RawName}" from file: {file_path}')
        elif type(InfoRaw) == pd.DataFrame:
            logger.info(f'RawName "{RawName}" from dataframe')
        else:
            raise ValueError(f'RawName "{RawName}" not supported with type: {type(InfoRaw)}')
        RawConfig['InfoRaw'] = InfoRaw
        RawName_to_RawConfig[RawName] = RawConfig

    # 6. get the df_HumanSelected: patients who have the corresponding record according to df_Human
    df_HumanSelected = get_df_HumanSelected_from_OneCohortRecArgs(rec_config, 
                                                                  RawName_to_RawConfig, 
                                                                  OneCohort_config, 
                                                                  df_Human,
                                                                  base_config)
    
    logger.info(f'df_HumanSelected shape: {df_HumanSelected.shape}')

    # 7. get the df_HumanGroup: group the df_HumanSelected based on the index_group
    #    sometimes df_HumanSelected is too large, so we want to separate it into smaller groups to do process. 
    ds_rec_list = []
    for index_group, df_HumanGroup in df_HumanSelected.groupby('index_group'): 
        logger.info(f'current index_group: {index_group} ...')

        # ---------------------- this is the core part of the pipeline ----------------------
        # 7.1 get the df_HumanRawRec, this function can be used independently to get the raw df_HumanRawRec. 
        df_HumanRawRec = get_HumanRawRec_for_HumanGroup(df_HumanGroup, RawName_to_RawConfig, RawName_to_dfRaw, base_config)

        # update df_HumanRawRec based on df_HumanSelected
        index = df_HumanRawRec[RawRootID].isin(df_HumanSelected[RawRootID].to_list())
        df_HumanRawRec = df_HumanRawRec[index].reset_index(drop = True)
        logger.info(f'current df_HumanRawRec: {df_HumanRawRec.shape} ...')

        # 7.2 get the df_HumanRecAttr: This is the pipeline A. 

        df_HumanRecAttr = RawRec_to_RecAttr_fn(df_HumanRawRec, df_Human, base_config, rec_config, attr_cols)
        
        logger.info(f'current df_HumanRecAttr: {df_HumanRecAttr.shape} ...')
        # ---------------------------------------

        if len(df_HumanRecAttr) == 0: continue 
        ds_HumanRecAttr = datasets.Dataset.from_pandas(df_HumanRecAttr)
        ds_rec_list.append(ds_HumanRecAttr)
        del df_HumanRecAttr

    ds_rec = datasets.concatenate_datasets(ds_rec_list)
    logger.info(f'current ds_rec for RecName {RecName}: {len(ds_rec)} ...')
    return ds_rec


def pipeline_for_FldTkn(ds_rec, fldtkn_config):
    # small phi
    pypath = fldtkn_config['pypath']
    logger.info(f'load fldtkn pipeline from: {pypath} ...')
    module = load_module_variables(pypath)
    tokenizer_fn = module.MetaDict['tokenizer_fn']

    idx2tkn = module.MetaDict['idx2tkn']
    # possible other information
    if 'column_to_top_values' in module.MetaDict:
        fldtkn_config['column_to_top_values'] = module.MetaDict['column_to_top_values']
    if 'item_to_configs' in module.MetaDict:
        fldtkn_config['item_to_configs'] = module.MetaDict['item_to_configs']

    # convert it to a mapping function.
    tokenizer_transform = Tokenizer_Transform(tokenizer_fn, idx2tkn, fldtkn_config)
    batch_size = fldtkn_config.get('batch_size', 10000)
    num_proc = fldtkn_config.get('num_proc', 4)
    num_proc = min(num_proc, len(ds_rec))
    ds_rec = ds_rec.map(tokenizer_transform, batched = True, num_proc=num_proc, batch_size=batch_size)
    logger.info(f'ds_rec.column_names: {ds_rec.column_names} ...')
    return ds_rec 


def pipeline_record(record_to_recfldtkn_list, 
                    OneCohort_config,
                    rft_config, 
                    df_Human, 
                    RawName_to_dfRaw, 
                    load_from_disk = False, 
                    reuse_old_rft = False, 
                    save_to_disk = False):
    
    RecName_to_RecConfig = {}
    RecName_to_dsRec = {}
    RecName_to_dsRecInfo = {}

    # cohort information
    cohort_label = OneCohort_config['cohort_label']
    cohort_name = OneCohort_config['cohort_name']

    # rft config
    base_config = rft_config['base_config']
    SPACE = base_config['SPACE']
    for RecordName, recfldtkn_list in record_to_recfldtkn_list.items():
        logger.info(f'RecordName: {RecordName}')
        RootID = base_config['RootID']
        
        ################################################################
        # data_folder = os.path.join(cohort_args['hfds_folder'], RecordName)
        cohort_full_name = f'{cohort_label}-{cohort_name}'
        data_folder = os.path.join(SPACE['DATA_RFT'], cohort_full_name,  RecordName)
        ################################################################
        
        # rec_config = load_record_args(RecordName, base_config, use_inference)
        rec_config = rft_config['rec_configs'][RecordName]
        rec_config['Name'] = RecordName

        # RecName Part.
        if load_from_disk == True and os.path.exists(data_folder+'_data'):
            logger.info(f'Load from disk: {data_folder} ...')
            ds_rec = datasets.load_from_disk(data_folder + '_data')
            ds_rec_info = datasets.load_from_disk(data_folder + '_info')
        else:
            ds_rec = pipeline_for_RecName(rec_config,
                                          OneCohort_config,  
                                          rft_config,
                                          df_Human, 
                                          RawName_to_dfRaw,
                                          RecName_to_dsRec)
        
        assert ds_rec is not None 
        # if ds_rec is None: continue

        # 2. for the fldtkn part
        for fldtkn in recfldtkn_list:
            # fldtkn_config = load_fldtkn_args(RecordName, fldtkn, cohort_args)
            fldtkn_config = rft_config['fldtkn_configs'][fldtkn]
            fldtkn_config['Name'] = fldtkn
            logger.info(f'fldtkn: {fldtkn}')
            columns = [i for i in ds_rec.column_names if 'Tkn' in i]
            if reuse_old_rft == True and fldtkn_config['Name']+'_tknidx' in columns: 
                logger.info(f'fldtkn: {fldtkn} already exists in the ds_rec ...')
                continue 
            ds_rec = pipeline_for_FldTkn(ds_rec, fldtkn_config)

        # 3. for the df_rec_info part. 
        RecDT = rec_config.get('RecDT', None) # ['RecDT']
        if RecDT and RecDT in ds_rec.column_names:
            df_rec = ds_rec.select_columns([RootID, RecDT]).to_pandas()
        else:
            df_rec = ds_rec.select_columns([RootID]).to_pandas()
        ds_rec_info_generator = DS_Rec_Info_Generator(df_rec, RootID, RecDT)
        ds_rec_info = datasets.Dataset.from_generator(ds_rec_info_generator)
        del df_rec 

        # 4. place the ds_rec to the dictionary
        if save_to_disk == True: 
            logger.info(f'Save ds_rec to:      {data_folder}_data')
            logger.info(f'Save ds_rec_info to: {data_folder}_info')
            try:
                ds_rec.save_to_disk(data_folder + '_data')
                ds_rec_info.save_to_disk(data_folder + '_info')
            except Exception as e:
                logger.info(f'Error in saving: {e}')
                
                # save to another folder
                ds_rec.save_to_disk(data_folder + '_data_tmp')
                ds_rec_info.save_to_disk(data_folder + '_info_tmp')
                
                # release the memory
                del ds_rec, ds_rec_info

                # remove the old folder
                shutil.rmtree(data_folder + '_data')
                shutil.rmtree(data_folder + '_info')

                # rename the new folder 
                os.rename(data_folder + '_data_tmp', data_folder + '_data')
                os.rename(data_folder + '_info_tmp', data_folder + '_info')
                
                # load the new folder
                ds_rec = datasets.load_from_disk(data_folder + '_data')
                ds_rec_info = datasets.load_from_disk(data_folder + '_info')

        RecName_to_RecConfig[RecordName] = rec_config
        RecName_to_dsRec[RecordName] = ds_rec
        RecName_to_dsRecInfo[RecordName] = ds_rec_info

    results = {
        'RecName_to_RecConfig': RecName_to_RecConfig,
        'RecName_to_dsRec': RecName_to_dsRec,
        'RecName_to_dsRecInfo': RecName_to_dsRecInfo
    }
    return results
    

def pipeline_from_dfRaw_to_dsRec(PipelineInfo, 
                                 OneCohort_config, 
                                 rft_config, 
                                 RawName_to_dfRaw, 
                                 load_from_disk = False, 
                                 reuse_old_rft = False, 
                                 save_to_disk = False):

    # Step 0: Basic Information
    cohort_label = OneCohort_config['cohort_label']
    cohort_name = OneCohort_config['cohort_name']
    base_config = rft_config['base_config'] 
    SPACE = base_config['SPACE']

    # Step 1: df_Human
    path_dfHuman = os.path.join(SPACE['DATA_RFT'], 
                                f'{cohort_label}-{cohort_name}', 
                                base_config['RecName'] + '_data')
    if load_from_disk == True and os.path.exists(path_dfHuman):
        logger.info(f'Load from disk: {path_dfHuman} ...')
        df_Human, _ = load_ds_rec_and_info(base_config['RecName'], base_config)
    else:
        df_Human = get_CohortLevel_df_Human2RawRecNum(OneCohort_config, 
                                                      rft_config, 
                                                      RawName_to_dfRaw)
    if save_to_disk == True: 
        logger.info(f'Save df_Human to: {path_dfHuman}')
        df_Human.save_to_disk(path_dfHuman)
        
    # Step 2: ds_Rec - prepare information
    RecName_to_FldTkn_list = {}
    for FldTkn in PipelineInfo['FldTknList']:
        RecName = FldTkn.split('-')[0]
        if RecName not in RecName_to_FldTkn_list: 
            RecName_to_FldTkn_list[RecName] = []
        # FldTkn = FldTkn + 'Tkn'
        RecName_to_FldTkn_list[RecName].append(FldTkn)

    record_to_recfldtkn_list = {}
    for recname in PipelineInfo['RecNameList']:
        recfldtkn_list = [] if recname not in RecName_to_FldTkn_list else RecName_to_FldTkn_list[recname]
        record_to_recfldtkn_list[recname] = sorted(recfldtkn_list)

    # Step : ds_Rec - pipeline_record
    data_RecName_to_DsRec = pipeline_record(record_to_recfldtkn_list, 
                                            OneCohort_config,
                                            rft_config, 
                                            df_Human, 
                                            RawName_to_dfRaw, 
                                            load_from_disk = load_from_disk, 
                                            reuse_old_rft = reuse_old_rft, 
                                            save_to_disk = save_to_disk
                                            )
    
    return df_Human, data_RecName_to_DsRec


# ------------------------------------- deprecated -------------------------------------

def get_and_save_vocab_from_idx2tkn(idx2tkn, FldTknName, folderv = None,  **kwargs):
    tkn2idx = {v:k for k, v in enumerate(idx2tkn)}
    tkn2fld = {v:FldTknName for k, v in enumerate(idx2tkn)}
    Vocab = pd.Series({'idx2tkn': idx2tkn, 'tkn2idx': tkn2idx, 'tkn2fld': tkn2fld})
    if folderv is not None: 
        path = os.path.join(folderv, FldTknName + '.p')
        Vocab.to_pickle(path)
    return Vocab # print(path)


def tokenizer_dfHumanRecAttr(dfHumanRecAttr, RootID, RecID, 
                            FldTknName, tokenizer_fn, Vocab, 
                            fldtkn_args, use_tknidx = False):
    df = dfHumanRecAttr.copy()
    df[FldTknName] = df.apply(lambda rec: tokenizer_fn(rec, fldtkn_args), axis = 1)
    for i in ['tkn', 'wgt']: 
        df[f'{FldTknName}_{i}'] = df[FldTknName].apply(lambda x: x[i])
    df = df.drop(columns = [FldTknName])

    cols = [RecID, FldTknName + '_tkn', FldTknName + '_wgt']
    cols = [RootID] + cols if RootID not in cols else cols

    if use_tknidx:
        tkn2idx = Vocab['tkn2idx']
        df[FldTknName + '_tkn'] = df[FldTknName + '_tkn'].apply(lambda x: [tkn2idx[i] for i in x])
        df = df[cols]
        df = df.rename(columns = {FldTknName + '_tkn': FldTknName + '_tknidx'})
    else:
        df = df[cols]
    return df


def get_cohort_level_record_number_counts(cohort_name, 
                                          cohort_label, 
                                          base_config, 
                                          filepath_to_rawdf = None):
    
    # Section: Load the cohort information
    ###############################
    # cohort_args = base_config
    pypath = base_config['pypath']
    module = load_module_variables(pypath)
    selected_source_file_suffix_list = module.selected_source_file_suffix_list
    excluded_raw_table_names = module.excluded_raw_table_names
    get_tablename_from_raw_filename = module.get_tablename_from_raw_filename
    get_rawrootid_from_raw_table_column = module.get_rawrootid_from_raw_table_column
    ###############################

    RawRootID = base_config['RawRootID']
    RootID = base_config['RootID']  
    RootIDLength = base_config['RootIDLength']
    chunk_size = 100000

    ################## CohortInfo ##################
    OneCohort_config = base_config['CohortInfo'][cohort_name]
    FolderPath = OneCohort_config['FolderPath']
    ################################################
    logger.info(f'=======cohort_label-{cohort_label}: cohort_name-{cohort_name}=======')
    logger.info(OneCohort_config)
    
    # Section: Update the filepath_to_rawdf
    if filepath_to_rawdf is None: 
        file_list = [i for i in os.listdir(FolderPath) if i.split('.')[-1] in selected_source_file_suffix_list]
        fullfile_list = [os.path.join(FolderPath, i) for i in file_list]
        logger.info(f'{FolderPath} <-- FolderPath')
        logger.info(f'{len(fullfile_list)} <--- fullfile_list')
        filepath_to_rawdf = {filepath: None for filepath in fullfile_list}

    # Section: Collect the results (RecNum) from the raw data. 
    L = []
    for file_path, rawdf in filepath_to_rawdf.items():
        if type(rawdf) == pd.DataFrame:
            if len(rawdf) == 0: continue
            result = read_column_value_counts_by_chunk(RawRootID, 
                                                       chunk_size, 
                                                       file_path, 
                                                       get_rawrootid_from_raw_table_column, 
                                                       get_tablename_from_raw_filename,
                                                       rawdf)
            logger.info(f"'{file_path}' # {result.shape}")
        
        elif file_path.split('.')[-1] == 'csv':
            if os.stat(file_path).st_size == 0: 
                logger.info(f"'{file_path}' # emtpy file"); continue
            try:
                result = read_column_value_counts_by_chunk(RawRootID, 
                                                           chunk_size, 
                                                           file_path, 
                                                           get_rawrootid_from_raw_table_column, 
                                                           get_tablename_from_raw_filename,
                                                           rawdf)
                logger.info(f"'{file_path}' # {result.shape}")
            except Exception as e:
                logger.info(f"'{file_path}' # error file. with error as {e}"); continue
        
        else:
            if file_path.split('.')[-1] == 'parquet':
                rawdf = pd.read_parquet(file_path)
            elif file_path.split('.')[-1] == 'p':
                rawdf = pd.read_pickle(file_path)
            else:
                raise ValueError(f'file type not supported: {file_path}')
            
            result = read_column_value_counts_by_chunk(RawRootID, 
                                                       chunk_size, 
                                                       file_path, 
                                                       get_rawrootid_from_raw_table_column, 
                                                       get_tablename_from_raw_filename,
                                                       rawdf)
            
            logger.info(f"'{file_path}' # {result.shape}")
            
        L.append(result)
    logger.info(f'{len(L)} <---- types of dfRec so far')
    df_all = pd.concat(L, ignore_index=True)
    df_pivot = df_all.pivot(index=RawRootID, columns='RecName', values='RecNum').reset_index()
    
    # Section: Filtering the df_Human who does not have any records. 
    recname_cols = [i for i in df_pivot.columns if i != RawRootID]
    # -- only consider the records that is in the included_cols. 
    included_cols = [i for i in recname_cols if i not in excluded_raw_table_names]
    rec_count = df_pivot[included_cols].sum(axis = 1)
    df_Human = df_pivot[rec_count > 0].reset_index(drop = True)
    df_Human['TotalRecNum'] = df_Human[included_cols].sum(axis = 1)
    logger.info(len(df_Human))

    # Section: Adding and Updating the RootID. 
    CohortLabel = OneCohort_config['cohort_label']
    df_Human[RootID] = range(1, len(df_Human) + 1)
    df_Human[RootID] = df_Human[RootID].apply(lambda x: int(str(CohortLabel) + str(x).zfill(RootIDLength)))
    df_Human['CohortLabel'] = CohortLabel
    cols = ['PID'] + [i for i in df_Human.columns if i not in ['PID']]
    df_Human = df_Human[cols].reset_index(drop = True)

    return df_Human

