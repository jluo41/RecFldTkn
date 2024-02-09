import pandas as pd

import numpy as np

idx2tkn = ['zip3-None', 'AboveBachelor', 'AboveBachelor-median', 'AboveBachelor-median_None', 'AboveBachelor-var', 'AboveBachelor-var_None', 'AboveBachelor_None', 'AboveHighSchool', 'AboveHighSchool-median', 'AboveHighSchool-median_None', 'AboveHighSchool-var', 'AboveHighSchool-var_None', 'AboveHighSchool_None', 'CohabitingcoupleHousehold', 'CohabitingcoupleHousehold-median', 'CohabitingcoupleHousehold-median_None', 'CohabitingcoupleHousehold-var', 'CohabitingcoupleHousehold-var_None', 'CohabitingcoupleHousehold_None', 'EnglishOnly', 'EnglishOnly-median', 'EnglishOnly-median_None', 'EnglishOnly-var', 'EnglishOnly-var_None', 'EnglishOnly_None', 'FemaleHouseholder', 'FemaleHouseholder-median', 'FemaleHouseholder-median_None', 'FemaleHouseholder-var', 'FemaleHouseholder-var_None', 'FemaleHouseholder_None', 'HouseHold', 'HouseHold-median', 'HouseHold-median_None', 'HouseHold-var', 'HouseHold-var_None', 'HouseHold_None', 'MaleHousegolder', 'MaleHousegolder-median', 'MaleHousegolder-median_None', 'MaleHousegolder-var', 'MaleHousegolder-var_None', 'MaleHousegolder_None', 'MarriedCoupleHouseHold', 'MarriedCoupleHouseHold-median', 'MarriedCoupleHouseHold-median_None', 'MarriedCoupleHouseHold-var', 'MarriedCoupleHouseHold-var_None', 'MarriedCoupleHouseHold_None', 'MarriedFemale', 'MarriedFemale-median', 'MarriedFemale-median_None', 'MarriedFemale-var', 'MarriedFemale-var_None', 'MarriedFemale_None', 'MarriedMale', 'MarriedMale-median', 'MarriedMale-median_None', 'MarriedMale-var', 'MarriedMale-var_None', 'MarriedMale_None', 'WithInternet', 'WithInternet-median', 'WithInternet-median_None', 'WithInternet-var', 'WithInternet-var_None', 'WithInternet_None', 'WomenHadBirthIn12Mon', 'WomenHadBirthIn12Mon-median', 'WomenHadBirthIn12Mon-median_None', 'WomenHadBirthIn12Mon-var', 'WomenHadBirthIn12Mon-var_None', 'WomenHadBirthIn12Mon_None']

def tokenizer_fn(rec, fldtkn_args):
    df_db = fldtkn_args['external_source']
    # df_db = fldtkn_args['df_zipdb_social']
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