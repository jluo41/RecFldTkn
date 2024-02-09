import pandas as pd

import numpy as np

selected_source_file_suffix_list = ['csv']

excluded_cols = []

def get_id_column(columns):
    if 'patient_id_encoded' in columns: id_column = 'patient_id_encoded' 
    return id_column


def get_tablename_from_file(file_path):
    name = file_path.split('/')[-1].split('_df_')[0]
    return name


def read_column_value_counts_by_chunk(RawRootID, chunk_size, file_path, rawdf = None):
    if type(rawdf) != pd.DataFrame:
        columns = pd.read_csv(file_path, nrows=0).columns
    else:
        columns = rawdf.columns 
    id_column = get_id_column(columns)

    if type(rawdf) == pd.DataFrame:
        result = rawdf[id_column].value_counts()
    else:
        li = [chunk[id_column].value_counts() for chunk in pd.read_csv(file_path, 
                                                                       usecols = [id_column], 
                                                                       chunksize=chunk_size, 
                                                                       low_memory=False)]
        result = pd.concat(li)
        result = result.groupby(result.index).sum()

    name = get_tablename_from_file(file_path)
    result = result.reset_index().rename(columns = {'count': 'RecNum', id_column: RawRootID})
    result['RecName'] = name
    return result


MetaDict = {
	"selected_source_file_suffix_list": selected_source_file_suffix_list,
	"excluded_cols": excluded_cols,
	"get_id_column": get_id_column,
	"get_tablename_from_file": get_tablename_from_file,
	"read_column_value_counts_by_chunk": read_column_value_counts_by_chunk
}