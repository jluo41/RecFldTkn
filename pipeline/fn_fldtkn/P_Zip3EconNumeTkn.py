import pandas as pd

import numpy as np

idx2tkn = ['zip3-None', 'AllWithInsurance', 'AllWithInsurance-median', 'AllWithInsurance-var', 'AllWithInsurance-var_None', 'CommutingWithCar', 'CommutingWithCar-median', 'CommutingWithCar-var', 'CommutingWithCar-var_None', 'ComuntingMinutes', 'ComuntingMinutes-median', 'ComuntingMinutes-median_None', 'ComuntingMinutes-var', 'ComuntingMinutes-var_None', 'ComuntingMinutes_None', 'Employed', 'Employed-median', 'Employed-var', 'Employed-var_None', 'FemalesEmployed', 'FemalesEmployed-median', 'FemalesEmployed-var', 'FemalesEmployed-var_None', 'FoodSNAP', 'FoodSNAP-median', 'FoodSNAP-var', 'FoodSNAP-var_None', 'HousehouldMedianIncome', 'HousehouldMedianIncome-median', 'HousehouldMedianIncome-median_None', 'HousehouldMedianIncome-var', 'HousehouldMedianIncome-var_None', 'HousehouldMedianIncome_None', 'LowIncome', 'LowIncome-median', 'LowIncome-var', 'LowIncome-var_None', 'MedianFamilyIncome', 'MedianFamilyIncome-median', 'MedianFamilyIncome-median_None', 'MedianFamilyIncome-var', 'MedianFamilyIncome-var_None', 'MedianFamilyIncome_None', 'PctFamilyBelowPoverty-median_None', 'PctFamilyBelowPoverty-var_None', 'PctFamilyBelowPoverty_None', 'Unemployed', 'Unemployed-median', 'Unemployed-var', 'Unemployed-var_None', 'WithPrivateInsurance', 'WithPrivateInsurance-median', 'WithPrivateInsurance-var', 'WithPrivateInsurance-var_None', 'WithPublicInsurance', 'WithPublicInsurance-median', 'WithPublicInsurance-var', 'WithPublicInsurance-var_None']

def tokenizer_fn(rec, fldtkn_args):
    # df_db = fldtkn_args['df_zipdb_econ']
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