import pandas as pd

import numpy as np

idx2tkn = ['zip3-None', 'Above18YearsOld', 'Above18YearsOld-median', 'Above18YearsOld-var', 'Above18YearsOld-var_None', 'Above65YearsOld', 'Above65YearsOld-median', 'Above65YearsOld-var', 'Above65YearsOld-var_None', 'Asian', 'Asian-median', 'Asian-var', 'Asian-var_None', 'BlackPeople', 'BlackPeople-median', 'BlackPeople-var', 'BlackPeople-var_None', 'Female', 'Female-median', 'Female-var', 'Female-var_None', 'Male', 'Male-median', 'Male-var', 'Male-var_None', 'White', 'White-median', 'White-var', 'White-var_None']

def tokenizer_fn(rec, fldtkn_args):
    df_db = fldtkn_args['external_source']
    # df_db = cohort_args['df_zipdb_demo']
    try:
        zip3 = str(int(rec['patient_zipcode_3']))
    except:
        zip3 = str(rec['patient_zipcode_3'])

    if zip3 not in df_db['Zip3'].to_list():
        return {'tkn': ['zip3-None'], 'wgt':[1]}
    
    row = df_db[df_db['Zip3'] == zip3].iloc[0].to_dict()
        
    tkn_col = [i for i in df_db.columns if 'tkn' in i][0]
    wgt_col = [i for i in df_db.columns if 'wgt' in i][0]
    output = {'tkn': row[tkn_col], 'wgt': row[wgt_col]}

    return output


MetaDict = {
	"idx2tkn": idx2tkn,
	"tokenizer_fn": tokenizer_fn
}