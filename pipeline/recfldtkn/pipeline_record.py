import os
import logging
import datasets
import pandas as pd
from functools import reduce
from .configfn import load_cohort_args, load_record_args, load_fldtkn_args
from .loadtools import load_module_variables, load_ds_rec_and_info 
# Setup basic configuration for logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)


# from recfldtkn.pipeline_record import get_cohort_level_record_number_counts

def get_cohort_level_record_number_counts(cohort_name, cohort_label, cohort_args, filepath_to_rawdf = None):
    ###############################
    pypath = cohort_args['pypath']
    module = load_module_variables(pypath)
    get_id_column = module.get_id_column
    read_column_value_counts_by_chunk = module.read_column_value_counts_by_chunk
    excluded_cols = module.excluded_cols
    pid_recnum_result_fn = read_column_value_counts_by_chunk
    selected_source_file_suffix_list = module.selected_source_file_suffix_list
    ###############################

    RawRootID = cohort_args['RawRootID']
    RootID = cohort_args['RootID']  
    RootIDLength = cohort_args['RootIDLength']
    cohort_config = cohort_args['CohortInfo'][cohort_name]
    FolderPath = cohort_config['FolderPath']
    chunk_size = 100000
    
    logger.info(f'=======cohort_label-{cohort_label}: cohort_name-{cohort_name}=======')
    logger.info(cohort_config)
    
    if filepath_to_rawdf is None: 
        file_list = [i for i in os.listdir(FolderPath) if i.split('.')[-1] in selected_source_file_suffix_list]
        fullfile_list = [os.path.join(FolderPath, i) for i in file_list]
        logger.info(f'{FolderPath} <-- FolderPath')
        logger.info(f'{len(fullfile_list)} <--- fullfile_list')
        filepath_to_rawdf = {filepath: None for filepath in fullfile_list}

    L = []
    for file_path, rawdf in filepath_to_rawdf.items():
        if type(rawdf) == pd.DataFrame:
            if len(rawdf) == 0: continue
            result = pid_recnum_result_fn(RawRootID, chunk_size, file_path, rawdf)
            logger.info(f"'{file_path}' # {result.shape}")
        elif file_path.split('.')[-1] == 'csv':
            if os.stat(file_path).st_size == 0: 
                logger.info(f"'{file_path}' # emtpy file"); continue
            try:
                result = pid_recnum_result_fn(RawRootID, chunk_size, file_path, rawdf)
                logger.info(f"'{file_path}' # {result.shape}")
            except:
                logger.info(f"'{file_path}' # error file"); continue
        else:
            if file_path.split('.')[-1] == 'parquet':
                rawdf = pd.read_parquet(file_path)
            elif file_path.split('.')[-1] == 'p':
                rawdf = pd.read_pickle(file_path)
            else:
                raise ValueError(f'file type not supported: {file_path}')
            result = pid_recnum_result_fn(RawRootID, chunk_size, file_path, rawdf)
            logger.info(f"'{file_path}' # {result.shape}")
            
        L.append(result)
    logger.info(f'{len(L)} <---- types of dfRec so far')
    df_all = pd.concat(L, ignore_index=True)
    df_pivot = df_all.pivot(index=RawRootID, columns='RecName', values='RecNum').reset_index()

    recname_cols = [i for i in df_pivot.columns if i != RawRootID]
    included_cols = [i for i in recname_cols if i not in excluded_cols]
    rec_count = df_pivot[included_cols].sum(axis = 1)
    
    df_Human = df_pivot[rec_count > 0].reset_index(drop = True)
    df_Human['TotalRecNum'] = df_Human[included_cols].sum(axis = 1)
    logger.info(len(df_Human))

    CohortLabel = cohort_config['cohort_label']
    df_Human[RootID] = range(1, len(df_Human) + 1)
    df_Human[RootID] = df_Human[RootID].apply(lambda x: int(str(CohortLabel) + str(x).zfill(RootIDLength)))
    df_Human['CohortLabel'] = CohortLabel
    cols = ['PID'] + [i for i in df_Human.columns if i not in ['PID']]
    df_Human = df_Human[cols].reset_index(drop = True)

    return df_Human


def get_parent_record_information(record_args, cohort_args, df_Human, record_to_ds_rec = {}):

    RootID = cohort_args['RootID']
    RawRootID = cohort_args['RawRootID']
    if record_args['ParentRecName'] is not None:
        # get the parent record information
        prt_record_args = load_record_args(record_args['ParentRecName'], cohort_args)

        # TO UPDATE: we can also load df_Prt from huggingface dataset. 
        # print(record_args['ParentRecName'], '<--- ParentRecName')
        # print([i for i in record_to_ds_rec])

        if record_args['ParentRecName'] in record_to_ds_rec:
            # print([i for i in record_to_ds_rec.keys()], '<--- record_to_ds_rec')
            ds_Prt = record_to_ds_rec[record_args['ParentRecName']]
        else:
            # parent_path = os.path.join(cohort_args['hfds_folder'], record_args['ParentRecName'])
            # ds_Prt = datasets.load_from_disk(parent_path)
            ds_Prt, _ = load_ds_rec_and_info(record_args['ParentRecName'], cohort_args)
        
        ds_Prt = ds_Prt.select_columns(prt_record_args['RecIDChain'] + prt_record_args['RawRecID'])
        df_Prt = ds_Prt.to_pandas()
        
        # update df_Human
        df_Human = df_Human[df_Human[RootID].isin(df_Prt[RootID].to_list())].reset_index(drop = True)
        
        # record_args['df_Prt'] = df_Prt
        # record_args['prt_record_args'] = prt_record_args
        # print(df_Human.shape, '<--- df_Human from the previous df_Prt')
    else: 
        # get the parent record information
        prt_record_args = {}
        prt_record_args['RawRecID'] = cohort_args['RawRootID']
        df_Prt = df_Human[[RootID, RawRootID]]
        # record_args['df_Prt'] = df_Human[[RootID, RawRootID]]
        # record_args['prt_record_args'] = prt_record_args
        df_Human = df_Human

    return prt_record_args, df_Prt, df_Human 


def get_df_HumanSelected_from_OneCohortRecArgs(df_Human, record_args, OneCohortRec_args, 
                                               cohort_args, filepath_to_rawdf = None):
    
    RootID = cohort_args['RootID']
    RawRootID = cohort_args['RawRootID']
    cohort_label = OneCohortRec_args['cohort_label']
    # cohort_name = OneCohortRec_args['cohort_name']
    idx_group_size = record_args['idx_group_size']

    RelatedColInHuman_list = [RawTable_args['RecNumColumn'] for RawTable, RawTable_args in OneCohortRec_args.items() 
                              if RawTable not in ['cohort_label', 'cohort_name']]
    dfx = df_Human[[RootID, RawRootID, 'CohortLabel'] + RelatedColInHuman_list]
    dfx = dfx[dfx['CohortLabel'] == cohort_label].reset_index(drop = True)

    dfx = dfx[(dfx[RelatedColInHuman_list]>0).mean(axis = 1) == 1].reset_index(drop = True)
    dfx['index_group'] = (dfx.index / idx_group_size).astype(int)
    df_HumanSelected = dfx

    return df_HumanSelected


def load_HumanRawTable(PatientIDValue_to_RecNum, # from df_Human
                       tablename, PatientIDName, # from data.
                       raw_data_path, raw_columns, 
                       chunk_size = 100000, df_rawrec = None, **kwargs):
    
    PatientID_list = [i for i in PatientIDValue_to_RecNum]
    total_RecNum = sum([v for k, v in PatientIDValue_to_RecNum.items()])

    if type(df_rawrec) == pd.DataFrame:
        df_HumanRec = df_rawrec
        df_HumanRec = df_HumanRec[df_HumanRec[PatientIDName].isin(PatientID_list)].reset_index(drop = True)
    else:
        df_HumanRec = pd.DataFrame()
        for chunk in pd.read_csv(raw_data_path, chunksize=chunk_size, low_memory=False):
            chunk = chunk[chunk[PatientIDName].isin(PatientID_list)]
            df_HumanRec = pd.concat([df_HumanRec, chunk])
            if len(df_HumanRec) == total_RecNum:
                break
            if len(df_HumanRec) > total_RecNum:
                raise ValueError(f'{tablename} more than given number {total_RecNum}')
        df_HumanRec = df_HumanRec.reset_index(drop = True)

    if type(raw_columns) == list: 
        df_HumanRec = df_HumanRec[raw_columns].reset_index(drop = True)

    return df_HumanRec


def get_HumanRawRec_for_HumanGroup(df_HumanGroup, OneCohortRec_args, RawRootID, filepath_to_rawdf = {}):
    L = []
    for RawTable, RawTable_args in OneCohortRec_args.items():
        if RawTable in ['cohort_label', 'cohort_name']: continue

        RelatedColInHuman = RawTable_args['RecNumColumn']
        PatientIDName = RawTable_args['RawRootID']
        PatientIDValue_to_RecNum = dict(zip(df_HumanGroup[RawRootID], df_HumanGroup[RelatedColInHuman]))
        
        raw_data_path = RawTable_args['raw_data_path']
        raw_columns = RawTable_args['raw_columns']
        chunk_size = RawTable_args.get('chunk_size', 100000)


        selected_filepath = None
        # print(filepath_to_rawdf, '<--- filepath_to_rawdf')
        for filepath, df_rawrec in filepath_to_rawdf.items():
            # print(filepath, '<--- filepath')
            # print(raw_data_path, '<--- raw_data_path')
            raw_file_name = os.path.basename(raw_data_path).split('.')[0]
            file_name = os.path.basename(filepath).split('.')[0]
            if raw_file_name == file_name:
                selected_filepath = filepath

        # print(selected_filepath, '<--- selected_filepath\n\n')
        if selected_filepath is not None:
            df_rawrec = filepath_to_rawdf[selected_filepath]
        else:
            df_rawrec = None 

        # print(df_rawrec, '<--- df_rawrec')
        # assert df_rawrec is not None
        df_HumanRawRec_of_RawTable = load_HumanRawTable(PatientIDValue_to_RecNum, 
                                                        RawTable, PatientIDName, 
                                                        raw_data_path, raw_columns, 
                                                        chunk_size, df_rawrec)
        
        df = df_HumanRawRec_of_RawTable.rename(columns = {PatientIDName: RawRootID})
        L.append(df)
    df_HumanRawRec = reduce(lambda left, right: pd.merge(left, right, on=RawRootID, how='outer'), L)
    return df_HumanRawRec



class Tokenizer_Transform:
    def __init__(self, tokenizer_fn, idx2tkn, fldtkn_args):
        self.tokenizer_fn = tokenizer_fn
        self.idx2tkn = idx2tkn
        self.tkn2idx = {v:k for k, v in enumerate(idx2tkn)}
        self.fldtkn_args = fldtkn_args

    def __call__(self, examples):
        tokenizer_fn = self.tokenizer_fn
        df_rec = pd.DataFrame({k:v for k, v in examples.items()})
        df_output = pd.DataFrame([tokenizer_fn(rec, self.fldtkn_args) for idx, rec in df_rec.iterrows()])
        
        # these are the two new columns 
        df_output['tknidx'] = df_output['tkn'].apply(lambda x: [self.tkn2idx[i] for i in x])   
        df_output = df_output.drop(columns = ['tkn'])

        for k, v in df_output.to_dict('list').items():
            examples[self.fldtkn_args['Name'] + '_' + k] = v
        return examples



class DS_Rec_Info_Generator:
    def __init__(self, df_rec, RootID, RecDT):
        self.df_rec = df_rec
        self.RootID = RootID
        self.RecDT = RecDT

    def __call__(self):
        idx = 0
        for PID, df_rec_p in self.df_rec.groupby('PID'):
            d = {}
            d[f'{self.RootID}_idx'] = idx  
            d[self.RootID] = PID
            d['interval'] = min(df_rec_p.index), max(df_rec_p.index)
            if self.RecDT is not None:
                d['dates'] = [i.isoformat() for i in df_rec_p[self.RecDT].tolist()]
            idx = idx + 1
            yield d



def pipeline_for_a_record_type(record_args,
                               cohort_name, cohort_label,  
                               df_Human, cohort_args, 
                               record_to_ds_rec = {}, 
                               filepath_to_rawdf = {},
                               ):
    '''
    Pipeline A information

    Args:
        # --- record --- 
        record_args: the record name information. 

        # ---- OneCohort ----
        cohort_name: the cohort name.
        cohort_label: the cohort label. 

        # ---- cohort ----
        df_Human: the human information, for each patient, the record number for each raw table
        cohort_args: the all cohort information for the whole project. 
    '''

    # 1. get record_args: record_related information
    # record_args = load_record_args(RecName, cohort_args) <--- should be prepare outside. It comes from yaml file.
    # record_args's cohort information should containing cohort_name. 
    RootID = cohort_args['RootID']
    RawRootID = cohort_args['RawRootID']
    df_Human = df_Human.copy()

    # 2. get the pipeline basic function
    pypath = record_args['pypath']
    assert os.path.exists(pypath)
    logger.info(f'load recattr pipeline from: {pypath} ...')

    module = load_module_variables(pypath)
    RawRec_to_RecAttr_fn = module.RawRec_to_RecAttr_fn
    attr_cols = record_args['attr_cols']
    assert len(attr_cols) > 0
    logger.info(f'attr_cols: ...')
    logger.info(attr_cols)

    # 3. check the existence of parent record.
    #    and add the df_Prt to the record_args.
    #    and update df_Human based on df_Prt. 
    prt_record_args, df_Prt, df_Human = get_parent_record_information(record_args, cohort_args, 
                                                                      df_Human, record_to_ds_rec)

    # 4. select the patient cohort.
    df_Prt = df_Prt[df_Prt[RootID].astype(str).apply(lambda x: str(x)[0] == str(cohort_label))].reset_index(drop = True)
    df_Human = df_Human[df_Human[RootID].isin(df_Prt[RootID].to_list())].reset_index(drop = True)
    record_args['df_Prt'] = df_Prt
    record_args['prt_record_args'] = prt_record_args
    logger.info(f'df_Prt shape: {df_Prt.shape}')

    # 5. check the source data existence
    #    this part is a bit confusing. 
    source_path_not_existence_flag = 0
    OneCohortRec_args = record_args['CohortInfo'][cohort_name]
    OneCohortRec_args['cohort_name'] = cohort_name  
    OneCohortRec_args['cohort_label'] = cohort_label
    
    for tablename, tableinfo in OneCohortRec_args.items():
        if tablename in ['cohort_name', 'cohort_label']: continue
        raw_data_path = tableinfo['raw_data_path']
        logger.info(f'table info raw_data_path: {raw_data_path}')

        if not os.path.exists(raw_data_path): 
            matched_df_in_filepath_to_rawdf = 0
            for filepath in filepath_to_rawdf:
                # if filename in filepath:
                # logger.info(f'find the file in the filepath_to_rawdf')
                filename = os.path.basename(filepath).split('.')[0]
                raw_filename = os.path.basename(raw_data_path).split('.')[0]
                if filename == raw_filename: matched_df_in_filepath_to_rawdf = 1
            if matched_df_in_filepath_to_rawdf == 0:
                source_path_not_existence_flag += matched_df_in_filepath_to_rawdf
                logger.info(f'table info raw_data_path does not exist')

    if source_path_not_existence_flag > 0:
        logger.info(f'=== source_path_not_existence_flag: {source_path_not_existence_flag}')
        ds_rec = None 
        logger.info('return the empty ds_rec.')
        return ds_rec
    # print(CohortRec_args)

    # 6. get the df_HumanSelected: patients who have the corresponding record according to df_Human
    df_HumanSelected = get_df_HumanSelected_from_OneCohortRecArgs(df_Human, record_args, OneCohortRec_args, cohort_args)
    logger.info(f'df_HumanSelected shape: {df_HumanSelected.shape}')

    # 7. get the df_HumanGroup: group the df_HumanSelected based on the index_group
    #    sometimes df_HumanSelected is too large, so we want to separate it into smaller groups to do process. 
    ds_rec_list = []
    for index_group, df_HumanGroup in df_HumanSelected.groupby('index_group'): 
        logger.info(f'current index_group: {index_group} ...')

        # ---------------------- this is the core part of the pipeline ----------------------
        # 7.1 get the df_HumanRawRec
        #     this function can be used independently to get the raw df_HumanRawRec. 
        df_HumanRawRec = get_HumanRawRec_for_HumanGroup(df_HumanGroup, OneCohortRec_args, RawRootID, filepath_to_rawdf)
        index = df_HumanRawRec[RawRootID].isin(df_HumanSelected[RawRootID].to_list())
        df_HumanRawRec = df_HumanRawRec[index].reset_index(drop = True)
        logger.info(f'current df_HumanRawRec: {df_HumanRawRec.shape} ...')

        # 7.2 get the df_HumanRecAttr: This is the pipeline A. 
        df_HumanRecAttr = RawRec_to_RecAttr_fn(df_HumanRawRec, df_Human, cohort_args, record_args, attr_cols)
        logger.info(f'current df_HumanRecAttr: {df_HumanRecAttr.shape} ...')
        # ---------------------------------------

        if len(df_HumanRecAttr) == 0: continue 
        ds_HumanRecAttr = datasets.Dataset.from_pandas(df_HumanRecAttr)
        ds_rec_list.append(ds_HumanRecAttr)
        del df_HumanRecAttr

    ds_rec = datasets.concatenate_datasets(ds_rec_list)
    logger.info(f'current ds_rec: {len(ds_rec)} ...')
    # not yet save it. 
    return ds_rec


def pipeline_for_adding_fldtkn_to_ds_rec(ds_rec, fldtkn_args):
    # small phi
    pypath = fldtkn_args['pypath']
    logger.info(f'load fldtkn pipeline from: {pypath} ...')
    module = load_module_variables(pypath)
    tokenizer_fn = module.MetaDict['tokenizer_fn']

    idx2tkn = module.MetaDict['idx2tkn']
    # possible other information
    if 'column_to_top_values' in module.MetaDict:
        fldtkn_args['column_to_top_values'] = module.MetaDict['column_to_top_values']
    if 'item_to_configs' in module.MetaDict:
        fldtkn_args['item_to_configs'] = module.MetaDict['item_to_configs']

    # print(f'\n=================== process fldtkn: {FldTknName} ===================')
    tokenizer_transform = Tokenizer_Transform(tokenizer_fn, idx2tkn, fldtkn_args)
    
    ds_rec = ds_rec.map(tokenizer_transform, batched = True, num_proc=4, batch_size=10000)
    # logger.info(f'idx2tkn list: {idx2tkn} ...')
    logger.info(f'ds_rec.column_names: {ds_rec.column_names} ...')
    return ds_rec 

def pipeline_record(record_to_recfldtkn_list, 
                    cohort_name, cohort_label,  
                    df_Human, cohort_args, 
                    filepath_to_rawdf = {}, 
                    load_from_disk = False, 
                    reuse_old_rft = False, 
                    save_to_disk = False, 
                    use_inference = False):
    
    record_to_record_args = {}
    record_to_ds_rec = {}
    record_to_ds_rec_info = {}

    SPACE = cohort_args['SPACE']
    # print(record_to_recfldtkn_list, '<--- record_to_recfldtkn_list')
    for RecordName, recfldtkn_list in record_to_recfldtkn_list.items():
        logger.info(f'RecordName: {RecordName}')
        RootID = cohort_args['RootID']
        
        ################################################################
        # data_folder = os.path.join(cohort_args['hfds_folder'], RecordName)
        cohort_full_name = f'{cohort_label}-{cohort_name}'
        data_folder = os.path.join(SPACE['DATA_RFT'], cohort_full_name,  RecordName)
        ################################################################
        
        record_args = load_record_args(RecordName, cohort_args, use_inference)
        print(record_args['CohortInfo'])
        record_args['Name'] = RecordName

        # [0]. load from the disk or not
        if load_from_disk == True and os.path.exists(data_folder):
            logger.info(f'Load from disk: {data_folder} ...')
            ds_rec = datasets.load_from_disk(data_folder + '_data')
            ds_rec_info = datasets.load_from_disk(data_folder + '_info')
            record_to_record_args[RecordName] = record_args
            record_to_ds_rec[RecordName] = ds_rec
            record_to_ds_rec_info[RecordName] = ds_rec_info
            continue 
            
        # 1. for the record part 
        # print(RecordName, '<------------ RecordName')
        # print(record_to_ds_rec, '<--- record_to_ds_rec')
        # print(filepath_to_rawdf, '<--- record_to_ds_rec')

        ds_rec = pipeline_for_a_record_type(record_args, cohort_name, cohort_label, 
                                            df_Human, cohort_args, record_to_ds_rec, filepath_to_rawdf)
        
        assert ds_rec is not None 

        if ds_rec is None: continue

        # 2. for the fldtkn part
        for fldtkn in recfldtkn_list:
            fldtkn_args = load_fldtkn_args(RecordName, fldtkn, cohort_args)
            fldtkn_args['Name'] = fldtkn
            logger.info(f'fldtkn: {fldtkn}')
            columns = [i for i in ds_rec.column_names if 'Tkn' in i]
            if reuse_old_rft == True and fldtkn_args['Name']+'_tknidx' in columns: continue 
            ds_rec = pipeline_for_adding_fldtkn_to_ds_rec(ds_rec, fldtkn_args)

        # 3. for the df_rec_info part. 
        RecDT = record_args.get('RecDT', None) # ['RecDT']
        if RecDT and RecDT in ds_rec.column_names:
            # RecDT = record_args['RecDT']
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
            ds_rec.save_to_disk(data_folder + '_data')
            ds_rec_info.save_to_disk(data_folder + '_info')

        record_to_record_args[RecordName] = record_args
        record_to_ds_rec[RecordName] = ds_rec
        record_to_ds_rec_info[RecordName] = ds_rec_info

    results = {
        'record_to_record_args': record_to_record_args,
        'record_to_ds_rec': record_to_ds_rec,
        'record_to_ds_rec_info': record_to_ds_rec_info
    }
    return results
    

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