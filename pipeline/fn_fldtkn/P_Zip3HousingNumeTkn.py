import pandas as pd

import numpy as np

idx2tkn = ['zip3-None', 'HousingUnits', 'HousingUnits-median', 'HousingUnits-var', 'HousingUnits-var_None', 'OccupiedHousingUnits', 'OccupiedHousingUnits-median', 'OccupiedHousingUnits-var', 'OccupiedHousingUnits-var_None', 'TreeOrMoreVehicles', 'TreeOrMoreVehicles-median', 'TreeOrMoreVehicles-var', 'TreeOrMoreVehicles-var_None']

def tokenizer_fn(rec, fldtkn_args):
    # df_db = fldtkn_args['df_zipdb_housing']
    # df_db = cohort_args['df_zipdb_demo']
    df_db = fldtkn_args['external_source']
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