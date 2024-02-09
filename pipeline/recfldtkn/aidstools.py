import os
import pandas as pd
import numpy as np 
import datasets
from datasets import concatenate_datasets


def map_case_to_case_IOTVT_type(x, out_ratio, test_ratio, valid_ratio):
    if x['RandInOut'] < out_ratio:
        InOut = 'out'
    else:
        InOut = 'in'    
    adjusted_valid_ratio = valid_ratio / (1-test_ratio)
    if x['CaseRltLoc'] > 1 - test_ratio:
        TVT = 'test'
    else:
        if x['RandValidation'] < adjusted_valid_ratio:
            TVT = 'valid'
        else:
            TVT = 'train'
    InOutTVT = InOut + '_' + TVT    
    return InOutTVT


def generate_random_tags(df):
    RootID, ObsDT = 'PID', 'ObsDT'

    # down sample 
    df['RandDownSample'] = np.random.rand(len(df))

    # in&out
    df_P = df[[RootID]].drop_duplicates().reset_index(drop = True)
    df_P['RandInOut'] = np.random.rand(len(df_P))
    df = pd.merge(df, df_P)

    # test
    df['CaseLoc'] = df.groupby(RootID).cumcount()
    df = pd.merge(df, df[RootID].value_counts().reset_index())
    df['CaseRltLoc'] = df['CaseLoc'] /  df['count']
    # test other options
    df['RandTest'] = np.random.rand(len(df))

    # validation
    df['RandValidation'] = np.random.rand(len(df))

    df = df.drop(columns = ['CaseLoc', 'count']).reset_index(drop = True)
    df = df.sort_values('RandDownSample').reset_index(drop = True)
    return df


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