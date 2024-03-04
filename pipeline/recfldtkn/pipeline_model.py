import os
import json
import yaml
import logging
import pandas as pd
import numpy as np
import datasets
from datasets import concatenate_datasets
from .ckpd_obs import Ckpd_ObservationS
from .loadtools import load_module_variables
from .obsname import convert_case_observations_to_co_to_observation, get_RecNameList_and_FldTknList
from .loadtools import fetch_trigger_tools
from .loadtools import fetch_casetag_tools, fetch_casefilter_tools
from .loadtools import load_ds_rec_and_info

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s:%(asctime)s:(%(filename)s@%(lineno)d %(name)s)]: %(message)s')


def load_complete_PipelineInfo(datapoint_args, base_config, use_inference):

    case_observations_total = []
    for k, v in datapoint_args.items():
        if use_inference == True and 'Output' in k: continue 
        case_observations_total = case_observations_total + v['case_observations']

    _, co_to_CaseObsNameInfo = convert_case_observations_to_co_to_observation(case_observations_total)
    PipelineInfo = get_RecNameList_and_FldTknList(co_to_CaseObsNameInfo)
    PipelineInfo['FldTknList'] = [i+'Tkn' for i in PipelineInfo['FldTknList']]

    # 3. get record_sequence
    record_sequence = PipelineInfo['RecNameList']
    RecName_to_PrtRecName = base_config['RecName_to_PrtRecName']
    record_sequence_prt = [RecName_to_PrtRecName[recname] for recname in record_sequence if RecName_to_PrtRecName[recname] != 'None']
    # print(record_sequence_prt)
    new_records = [i for i in record_sequence_prt if i not in record_sequence]
    while len(new_records) > 0:
        record_sequence.extend(new_records)
        record_sequence_prt = [RecName_to_PrtRecName[recname] for recname in record_sequence if RecName_to_PrtRecName[recname] != 'None']
        new_records = [i for i in record_sequence_prt if i not in record_sequence]
    record_sequence = [recname for recname in base_config['RecName_Sequence'] if recname in PipelineInfo['RecNameList']]

    PipelineInfo['RecNameList'] = record_sequence # 
    return PipelineInfo


def get_Trigger_Cases(TriggerCaseMethod, 
                      cohort_label_list, 
                      base_config, 
                      SPACE, 
                      RecName_to_dsRec = {},
                      RecName_to_dsRecInfo = {}):
    
    Trigger_Tools = fetch_trigger_tools(TriggerCaseMethod, SPACE)
    case_id_columns = Trigger_Tools['case_id_columns']
    special_columns = Trigger_Tools['special_columns'] 
    TriggerRecName = Trigger_Tools['TriggerRecName']
    convert_TriggerEvent_to_Caseset = Trigger_Tools['convert_TriggerEvent_to_Caseset']
    ###########################
    if TriggerRecName in RecName_to_dsRec:
        ds_rec = RecName_to_dsRec[TriggerRecName]
    else:
        ds_rec, _ = load_ds_rec_and_info(TriggerRecName, base_config, cohort_label_list)
    df_case = convert_TriggerEvent_to_Caseset(ds_rec, case_id_columns, special_columns, base_config)
    ###########################
    return df_case


def convert_TriggerCases_to_LearningCases(df_case, 
                                          cohort_label_list,
                                          Trigger2LearningMethods, 
                                          base_config, 
                                          use_inference):
    
    CFQ_to_CaseFeatInfo = {}

    if use_inference == True:
        Trigger2LearningMethods = [i for i in Trigger2LearningMethods if i.get('type', None) != 'learning-only']
       
    # print(Trigger2LearningMethods)
    SPACE = base_config['SPACE']
    for method in Trigger2LearningMethods:
        if method['op'] == 'Tag':
            name = method['Name']
            logger.info(f'CaseTag: {name}')
            CaseTag_Tools = fetch_casetag_tools(name, SPACE)

            subgroup_columns = CaseTag_Tools['subgroup_columns']
            if 'InfoRecName' in CaseTag_Tools:
                InfoRecName = CaseTag_Tools['InfoRecName']
                ds_info, _ = load_ds_rec_and_info(InfoRecName, base_config, cohort_label_list)
            else:
                ds_info = None

            fn_case_tagging = CaseTag_Tools['fn_case_tagging']
            df_case = fn_case_tagging(df_case, ds_info, subgroup_columns, base_config)

        elif method['op'] == 'Filter':
            name = method['Name']
            logger.info(f'CaseFilter: {name}')
            CaseFilter_Tools = fetch_casefilter_tools(name, SPACE)
            fn_case_filtering = CaseFilter_Tools['fn_case_filtering']
            
            logger.info(f'Before Filter: {df_case.shape}')
            df_case = fn_case_filtering(df_case)
            logger.info(f'After Filter: {df_case.shape}')

        elif method['op'] == 'CFQ':
            name = method['Name']
            pypath = os.path.join(SPACE['CODE_FN'], 'fn_learning', f'{name}.py')
            module = load_module_variables(pypath)
            fn_casefeat_querying = module.fn_casefeat_querying
            df_case, CaseFeatInfo = fn_casefeat_querying(df_case, base_config)
            # CaseFeatName = CaseFeatInfo['CaseFeatName']
            CFQ_to_CaseFeatInfo[name] = CaseFeatInfo

        elif method['op'] == 'TagCF':
            name = method['Name']
            pypath = os.path.join(SPACE['CODE_FN'], 'fn_learning', f'{name}.py')
            module = load_module_variables(pypath)
            fn_case_tagging_on_casefeat = module.fn_case_tagging_on_casefeat
            
            CFQName = method['CFQName']
            CFQ_to_CaseFeatInfo = CFQ_to_CaseFeatInfo[CFQName]
            df_case = fn_case_tagging_on_casefeat(df_case, CaseFeatInfo)

        else:
            raise ValueError(f'Unknown method: {method}')

    return df_case


# Function to split DataFrame into chunks
def split_dataframe(df, chunk_size):
    chunks = []
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(df[i*chunk_size:(i+1)*chunk_size])
    return chunks


def generate_random_tags(df, RANDOM_SEED):
    np.random.seed(RANDOM_SEED)
    RootID, ObsDT = 'PID', 'ObsDT'

    # down sample 
    df['RandDownSample'] = np.random.rand(len(df))

    # in&out
    df_P = df[[RootID]].drop_duplicates().reset_index(drop = True)
    df_P['RandInOut'] = np.random.rand(len(df_P))
    df = pd.merge(df, df_P)

    # test
    df['CaseLocInP'] = df.groupby(RootID).cumcount()
    df = pd.merge(df, df[RootID].value_counts().reset_index())
    df['CaseRltLocInP'] = df['CaseLocInP'] /  df['count']
    # test other options
    df['RandTest'] = np.random.rand(len(df))

    # validation
    df['RandValidation'] = np.random.rand(len(df))

    df = df.drop(columns = ['CaseLocInP', 'count']).reset_index(drop = True)
    df = df.sort_values('RandDownSample').reset_index(drop = True)

    random_columns = ['RandDownSample', 'RandInOut', 'CaseRltLocInP', 'RandTest', 'RandValidation']
    return df, random_columns


def assign_caseSplitTag_to_dsCaseLearning(df_case_learning, 
                                          RANDOM_SEED, 
                                          downsample_ratio, out_ratio, 
                                          test_ratio, valid_ratio):

    df = df_case_learning 
    df_rs, random_columns = generate_random_tags(df, RANDOM_SEED)
    df_dsmp = df_rs[df_rs['RandDownSample'] <= downsample_ratio].reset_index(drop = True)

    df_dsmp['Out'] = df_dsmp['RandInOut'] < out_ratio
    df_dsmp['In'] = df_dsmp['RandInOut'] >= out_ratio
    assert df_dsmp[['Out', 'In']].sum(axis = 1).mean() == 1

    if 'tail' in str(test_ratio):
        TestSelector = 'CaseRltLocInP'
        test_ratio = float(test_ratio.replace('tail', ''))
        test_threshold = 1 - test_ratio
    elif type(test_ratio) != float and type(test_ratio) != int:
        TestSelector = 'ObsDT'
        test_threshold = pd.to_datetime(test_ratio)
    else:
        TestSelector = 'RandTest'
        test_threshold = 1 - test_ratio

    if 'tail' in str(valid_ratio):
        ValidSelector = 'CaseRltLocInP'
        valid_ratio = float(valid_ratio.replace('tail', ''))
        valid_threshold = 1 - valid_ratio
    elif type(valid_ratio) != float and type(valid_ratio) != int:
        ValidSelector = 'ObsDT'
        valid_threshold = pd.to_datetime(valid_ratio)
    else:
        ValidSelector = 'RandTest' 
        valid_threshold = 1 - valid_ratio
        
    df_dsmp['Test'] = df_dsmp[TestSelector] > test_threshold
    df_dsmp['Valid'] = (df_dsmp[ValidSelector] > valid_threshold) & (df_dsmp['Test'] == False)
    df_dsmp['Train'] = (df_dsmp['Test'] == False) & (df_dsmp['Valid'] == False)

    assert df_dsmp[['Train', 'Valid', 'Test']].sum(axis = 1).mean() == 1

    df_dsmp = df_dsmp.drop(columns = random_columns)
    return df_dsmp


# -------------------------------- deprecated --------------------------------

def map_case_to_IOTVT_split_type(x, out_ratio, test_ratio, valid_ratio):
    if x['RandInOut'] < out_ratio:
        InOut = 'out'
    else:
        InOut = 'in'    
    
    if 'tail' in str(test_ratio):
        TestSelector = 'CaseRltLocInP'
        test_ratio = float(test_ratio.replace('tail', ''))
        test_threshold = 1 - test_ratio
    elif type(test_ratio) != float and type(test_ratio) != int:
        TestSelector = 'ObsDT'
        test_threshold = pd.to_datetime(test_ratio)
    else:
        TestSelector = 'RandTest' 
        # like converting 0.1 to 0.9
        test_threshold = 1 - test_ratio

    if 'tail' in str(valid_ratio):
        ValidSelector = 'CaseRltLocInP'
        valid_ratio = float(valid_ratio.replace('tail', ''))
        valid_threshold = 1 - valid_ratio
    elif type(valid_ratio) != float and type(valid_ratio) != int:
        ValidSelector = 'ObsDT'
        valid_threshold = pd.to_datetime(valid_ratio)
    else:
        ValidSelector = 'RandTest' 
        valid_threshold = 1 - valid_ratio
    
    if x[TestSelector] > test_threshold:
        TVT = 'test'
    else:
        if x[ValidSelector] > valid_threshold:
            TVT = 'valid'
        else:
            TVT = 'train'
    InOutTVT = InOut + '_' + TVT    
    return InOutTVT


def get_dataset_from_set_selector(set_selector, dataset_name, idx_to_groupname_all):
    # for train_set 
    groupname_dict = idx_to_groupname_all.copy()

    ds_subset_name = set_selector.split(':')[0]
    # print(ds_subset_name)
    ds_subset_name_list_all = ['in_train', 'in_valid', 'in_test', 'out_test', 'out_train', 'out_valid']
    ds_subset_name_list = [i for i in ds_subset_name_list_all if ds_subset_name in i]

    if ':' in set_selector:
        groupname_types = set_selector.split(':')[-1].split('&')
        # print(groupname_types)
        for groupname_type in groupname_types:
            # print(groupname_type)
            groupname_dict = {k: v for k, v in groupname_dict.items() if groupname_type in v}

    # print('the number of group:', len(groupname_dict))

    L = []
    for ds_subset_name in ds_subset_name_list:
        for idx, groupname in groupname_dict.items():
            if 'CaseFolder' not in dataset_name:
                path = os.path.join(dataset_name + '-' + ds_subset_name, groupname)
                # print(path)
                if os.path.exists(path) == False: continue
                ds = datasets.load_from_disk(path)
                print(groupname, len(ds))
            else:
                path = os.path.join(dataset_name + '-' + ds_subset_name, groupname + '.p')
                # print(path)
                if os.path.exists(path) == False: continue
                df = pd.read_pickle(path)
                ds = datasets.Dataset.from_pandas(df)
                print(groupname, len(ds))
            L.append(ds)
    if len(L) > 0:
        ds = concatenate_datasets(L)
    else:
        ds = None 
    return ds


def get_caseset_to_observe(group_id, CaseFolder, case_id_columns, cohort_args):
    RootID = cohort_args['RootID']; ObsDT  = 'ObsDT'
    group_name_list = [i for i in os.listdir(CaseFolder) if '.p' in i]
    group_name_list = dict(sorted({int(i.split('_')[0]): i.replace('.p', '') for i in group_name_list}.items()))

    if group_id in group_name_list:
        group_name = group_name_list[group_id]
        df_case = pd.read_pickle(os.path.join(CaseFolder, f'{group_name}.p'))
        
        if len(case_id_columns) > 0: 
            df_case = df_case[case_id_columns].reset_index(drop=True)

        ds_case = datasets.Dataset.from_pandas(df_case)
    else:
        group_name = None 
        ds_case = None
    return group_name, ds_case


def get_ds_case_dict_from_settings(caseset_name, 
                                   splitset_name_list, 
                                   subgroup_id_list, 
                                   subgroup_filter_fn, 
                                   cohort_args,
                                   case_id_columns = []):
    # caseset_name = args.caseset_name
    # print(caseset_name)
    # splitset_name_list = update_args_to_list(args.splitset_name_list)
    # print(splitset_name_list)
    SPACE = cohort_args['SPACE']
    case_type_name_list = []
    for splitset_name in splitset_name_list:
        if 'rs' in splitset_name:
            case_type_name_list.append(caseset_name + '-' + splitset_name)
        else:
            case_type_name_list.append(caseset_name)
    
    # case_type_name_list
    # case_id_columns = update_args_to_list(args.case_id_columns) 

    d = {}
    for case_type in case_type_name_list:
        # print('\n========================')
        # print('case_type---->', case_type)

        # L = []
        for group_id in subgroup_id_list:
            # print('group_id------->', group_id)
            CaseFolder = os.path.join(SPACE['DATA_TASK'], 'CaseFolder', case_type)
            # print(CaseFolder)
            
            if os.path.exists(CaseFolder) == False:
                print(f'CaseFolder does not exist: {CaseFolder}')
                print('TODO: ---- in the future, the Case Folder should be automatically created.')
                continue

            group_name, ds_case = get_caseset_to_observe(group_id, CaseFolder, case_id_columns, cohort_args)
            d[f'{case_type}|{group_name}'] = ds_case

    ds_case_dict = datasets.DatasetDict(d)
    
    if subgroup_filter_fn == 'None': 
        ds_case_dict = ds_case_dict
    else:
        # TODO
        ds_case_dict = ds_case_dict.filter(subgroup_filter_fn)
        
    return ds_case_dict


def get_data_source_information(groupname_ids, 
                                groupname_selected_list,
                                case_tkn_name_list, 
                                CaseFolder, 
                                CaseObsFolder):

    casetkn_name_to_namepattern = {i.split(':')[0]:i.split(':')[-1] for i in case_tkn_name_list}
    print(casetkn_name_to_namepattern)

    L = []
    for idx, groupname in enumerate(groupname_selected_list):
        path = os.path.join(CaseFolder, groupname + '.p')
        # print(path)
        df = pd.read_pickle(path)
        case_num = len(df)
        d = {}
        d['group_id'] = int(groupname_ids[idx])
        d['group_name'] = groupname
        d['case_num'] = len(df)

        folder = os.path.join(CaseObsFolder, groupname)
        if os.path.exists(folder):
            obs_list = os.listdir(folder)
        else:
            obs_list = [] 

        if len(obs_list) == 0:
            print('For group: ', groupname)
            print('No obs folder: ', folder)
            continue 
            
        d['obs_list'] = obs_list
        for case_tkn_name, pattern in casetkn_name_to_namepattern.items():
            case_tkn_name_keys = pattern.split('*')
            obs_selected = [s for s in obs_list if all(key in s for key in case_tkn_name_keys)]
            d[case_tkn_name] = obs_selected
        L.append(d)

    df = pd.DataFrame(L).sort_values('group_id').reset_index(drop=True)
    total_case_num = df['case_num'].sum()
    print('total_case_num: ', total_case_num)
    # print(df)

    columns = ['group_id', 'group_name', 'case_num']
    print(df[columns])
    return df 


def get_ds_tkn_from_dfinfo(df, case_tkn_name_list, CaseObsFolder):
    casetkn_name_to_namepattern = {i.split(':')[0]:i.split(':')[-1] for i in case_tkn_name_list}
    print(casetkn_name_to_namepattern)
    RootID, ObsDT = 'PID', 'ObsDT'
    group_name_to_ds = {}
    for idx, row in df.iterrows():
        group_name = row['group_name']
        # ds_cat = None 
        ds_cat = {}
        for idx, case_tkn_name in enumerate(casetkn_name_to_namepattern):
            obs_selected = row[case_tkn_name]
            assert len(obs_selected) == 1   
            obs_name = obs_selected[0]
            # print(group_name, case_tkn_name, len(obs_selected))

            path = os.path.join(CaseObsFolder, group_name, obs_name)
            # print(path)
            ds = datasets.load_from_disk(path)
            columns = [i for i in ds.column_names if i not in [RootID, ObsDT]]  
            for column in columns:
                ds = ds.rename_column(column, case_tkn_name + '-' + column)
            if idx != 0: ds = ds.remove_columns([RootID, ObsDT])
            # print(ds)
            ds_cat[case_tkn_name] = ds

        ds_cat = datasets.concatenate_datasets([v for k, v in ds_cat.items()], axis=1)
        group_name_to_ds[group_name] = ds_cat

    ds_tknidx_dict = datasets.DatasetDict(group_name_to_ds)
    return ds_tknidx_dict


def adding_random_labels_to_ds(ds_tknidx, test_ratio):
    RootID, ObsDT = 'PID', 'ObsDT'
    np.random.seed(42)
    ds_pdt = ds_tknidx.select_columns([RootID, ObsDT])
    df_pdt = ds_pdt.to_pandas()

    df_pdt['ObsIdx'] = df_pdt.groupby(RootID).cumcount()
    df_pdt = pd.merge(df_pdt, df_pdt[RootID].value_counts().reset_index())
    df_pdt['ObsIdxTile'] = df_pdt['ObsIdx'] /  df_pdt['count']
    df_pdt['FutCaseTest'] = (df_pdt['ObsIdxTile'] >= (1- test_ratio)).astype(int) 

    df_pdt['RandNum'] = np.random.rand(len(df_pdt))
    df_P = df_pdt[['PID']].drop_duplicates().reset_index(drop = True)
    print(df_P.shape)
    print(df_pdt.shape)
    df_P['RandPNum'] = np.random.rand(len(df_P))
    df_pdt = pd.merge(df_pdt, df_P)
    print(df_pdt.shape)

    ds_tknidx = ds_tknidx.add_column('ObsIdxTile', df_pdt['ObsIdxTile'])
    ds_tknidx = ds_tknidx.add_column('RandNum', df_pdt['RandNum'])
    ds_tknidx = ds_tknidx.add_column('RandPNum', df_pdt['RandPNum'])
    ds_tknidx = ds_tknidx.add_column('FutCaseTest', df_pdt['FutCaseTest'])
    # print(ds_tknidx)
    return ds_tknidx


def conduct_downsample_inout_train_valid_test(ds_tknidx, downsample_ratio, out_ratio, test_ratio, valid_ratio):

    ds_tknidx = adding_random_labels_to_ds(ds_tknidx, test_ratio)

    print(f'\n================= Downsample Ratio is: {downsample_ratio} =================')
    # df_tknidx = ds_tknidx.to_pandas()
    # ds_tknidx_downsampled = ds_tknidx.sort('RandNum').select(range(0, int(len(ds_tknidx) * downsample_ratio)))
    ds_tknidx_downsampled = ds_tknidx.filter(lambda example: example['RandNum'] <= downsample_ratio)
    # ds_tknidx_downsampled = ds_tknidx_downsampled.flatten_indices() 
    # df_tknidx = df_tknidx.sort_values('RandNum').iloc[:int(len(df_tknidx) * downsample_ratio)].reset_index(drop=True)
    # ds_tknidx_downsampled = datasets.Dataset.from_pandas(df_tknidx)
    print(ds_tknidx_downsampled)

    print(f'\n================= Split In & Out: out_ratio-{out_ratio} =================')
    ds_tknidx_downsampled_PSorted = ds_tknidx_downsampled.sort('RandPNum')
    ds_tknidx_downsampled_in  = ds_tknidx_downsampled_PSorted.select(range(0, int(len(ds_tknidx_downsampled_PSorted) * (1-out_ratio))))
    ds_tknidx_downsampled_in = ds_tknidx_downsampled_in.flatten_indices()
    ds_tknidx_downsampled_out = ds_tknidx_downsampled_PSorted.select(range(int(len(ds_tknidx_downsampled_PSorted) * (1-out_ratio)), len(ds_tknidx_downsampled_PSorted)))
    ds_tknidx_downsampled_out = ds_tknidx_downsampled_out.flatten_indices()

    # print(ds_tknidx_downsampled_in)
    # print(ds_tknidx_downsampled_out)

    ds_dict = {
        'in': ds_tknidx_downsampled_in,
        'out': ds_tknidx_downsampled_out
    }

    ####### for in data process #######
    name = 'in'
    ds = ds_dict[name]
    test_size = ds.select_columns(['FutCaseTest']).to_pandas()['FutCaseTest'].sum()
    train_size = len(ds) - test_size
    # print('train_valid_size: ', train_size)
    # print('test_size: ', test_size) 

    ds = ds.sort('FutCaseTest')
    total_size = len(ds)
    if test_size == 0:
        test_dataset = None 
    else:
        test_dataset = ds.select(range(total_size - test_size, total_size))

    trainvalid_dataset = ds.select(range(0, total_size - test_size))
    train_valid_split = trainvalid_dataset.train_test_split(test_size=valid_ratio)
    train_dataset = train_valid_split['train']
    valid_dataset = train_valid_split['test']

    columns_to_drop = ['FutCaseTest', 'RandNum', 'RandPNum', 'ObsIdxTile']
    d = {
        'in_train': train_dataset, # .remove_columns(columns_to_drop), 
        'in_valid': valid_dataset, # .remove_columns(columns_to_drop), 
        'in_test':  test_dataset, # .remove_columns(columns_to_drop),
    }

    ####### for in data process #######
    name = 'out'
    ds = ds_dict[name]
    d['out_whole'] = ds# .remove_columns(['FutCaseTest', 'RandNum'])

    test_size = ds.select_columns(['FutCaseTest']).to_pandas()['FutCaseTest'].sum()
    # train_size = len(ds_tknidx_downsampled) - test_size
    ds_tknidx_downsampled_sorted = ds.sort('FutCaseTest')
    total_size = len(ds)
    test_dataset = ds_tknidx_downsampled_sorted.select(range(total_size - test_size, total_size))
    d['out_test'] = test_dataset # .remove_columns(['FutCaseTest', 'RandNum'])

    raw_datasets = datasets.DatasetDict({k: v for k, v in d.items() if v is not None})
    # print(raw_datasets)
    for name in raw_datasets.keys():
        raw_datasets[name] = raw_datasets[name].remove_columns(columns_to_drop)

    return raw_datasets


def get_dataset_from_set_selector(set_selector, dataset_name, idx_to_groupname_all):
    # for train_set 
    groupname_dict = idx_to_groupname_all.copy()

    ds_subset_name = set_selector.split(':')[0]
    # print(ds_subset_name)
    ds_subset_name_list_all = ['in_train', 'in_valid', 'in_test', 'out_test', 'out_train', 'out_valid']
    ds_subset_name_list = [i for i in ds_subset_name_list_all if ds_subset_name in i]

    if ':' in set_selector:
        groupname_types = set_selector.split(':')[-1].split('&')
        # print(groupname_types)
        for groupname_type in groupname_types:
            # print(groupname_type)
            groupname_dict = {k: v for k, v in groupname_dict.items() if groupname_type in v}

    # print('the number of group:', len(groupname_dict))

    L = []
    for ds_subset_name in ds_subset_name_list:
        for idx, groupname in groupname_dict.items():
            path = os.path.join(dataset_name + '-' + ds_subset_name, groupname)
            if os.path.exists(path) == False: continue
            # print(path)
            ds = datasets.load_from_disk(path)
            print(groupname, len(ds))
            # print([i for i in ds])
            # ds = {k: v for k, v in ds.items() if ds_subset_name in k}
            # if ds_subset_name not in ds: continue
            # if len(ds) == 0: continue 
            # if len(ds) == 0:
            #     ds_subset_name_dataset = [v for k, v in ds.items()][0]
            # else:
            #     ds_subset_name_dataset = concatenate_datasets([v for k, v in ds.items()])
            # print(groupname, len(ds_subset_name_dataset))
            L.append(ds)
    if len(L) > 0:
        ds = concatenate_datasets(L)
    else:
        ds = None 
    return ds



def load_model_args(model_checkpoint_name, base_config, inference_mode = False):
    SPACE = base_config['SPACE']
    full_model_path = os.path.join(SPACE['MODEL_REPO'], model_checkpoint_name)  
    model_args_path = os.path.join(full_model_path, f'model_args.json')
    with open(model_args_path, 'r') as json_file:
        model_args = json.load(json_file)
        
    # 1. get TriggerCaseMethod
    TriggerCaseMethod = model_args['TriggerCaseMethod']
    pypath = os.path.join(base_config['trigger_pyfolder'], f'{TriggerCaseMethod}.py')
    module = load_module_variables(pypath)
    model_args['TriggerRecName'] = module.TriggerRecName
    model_args['case_id_columns'] = module.case_id_columns
    model_args['special_columns'] = module.special_columns
    model_args['convert_TriggerEvent_to_Caseset'] = module.convert_TriggerEvent_to_Caseset

    # 2. get case_observations
    if inference_mode == True:
        case_observations = model_args['case_observations']
        case_observations = [co for co in case_observations if 'Af' not in co.split(':')[0] and 'Fut' not in co.split(':')[0]]
        model_args['case_observations'] = case_observations
    else:
        case_observations = model_args['case_observations']
    co_to_CaseObsName, co_to_CaseObsNameInfo = convert_case_observations_to_co_to_observation(case_observations)
    PipelineInfo = get_RecNameList_and_FldTknList(co_to_CaseObsNameInfo)
    
    # 3. get record_sequence
    record_sequence = PipelineInfo['RecNameList']
    RecName_to_PrtRecName = base_config['RecName_to_PrtRecName']
    record_sequence_prt = [RecName_to_PrtRecName[recname] for recname in record_sequence if RecName_to_PrtRecName[recname] != 'None']
    # print(record_sequence_prt)
    new_records = [i for i in record_sequence_prt if i not in record_sequence]
    while len(new_records) > 0:
        record_sequence.extend(new_records)
        record_sequence_prt = [RecName_to_PrtRecName[recname] for recname in record_sequence if RecName_to_PrtRecName[recname] != 'None']
        new_records = [i for i in record_sequence_prt if i not in record_sequence]
    record_sequence = [recname for recname in base_config['RecName_Sequence'] if recname in PipelineInfo['RecNameList']]

    PipelineInfo['RecNameList'] = record_sequence # [recname for recname in base_config['RecNameList'] if recname in PipelineInfo['RecNameList']]
    PipelineInfo['FldTknList'] = sorted([fldtkn + 'Tkn' for fldtkn in PipelineInfo['FldTknList']])
    model_args['record_sequence'] = record_sequence
    model_args['PipelineInfo'] = PipelineInfo
    model_args['co_to_CaseObsName'] = co_to_CaseObsName
    model_args['co_to_CaseObsNameInfo'] = co_to_CaseObsNameInfo


    # 4. update the path

    path_list = [i for i in model_args if '_path' in i]
    for path_name in path_list:
        if full_model_path in model_args[path_name]: continue 
        model_args[path_name] = os.path.join(full_model_path, model_args[path_name])

    model_args['full_model_path'] = full_model_path
    return model_args

