import pandas as pd

import numpy as np

column_to_top_values = {'patient_age_bucket': ['51-60', '61-70', '41-50', '71-80', '31-40', '18-30', '81+'], 'patient_gender': ['F', 'M', 'U']}

idx2tkn = ['patient_age_bucket_unk', 'patient_age_bucket_minor', 'patient_age_bucket_51-60', 'patient_age_bucket_61-70', 'patient_age_bucket_41-50', 'patient_age_bucket_71-80', 'patient_age_bucket_31-40', 'patient_age_bucket_18-30', 'patient_age_bucket_81+', 'patient_gender_unk', 'patient_gender_minor', 'patient_gender_F', 'patient_gender_M', 'patient_gender_U']

def tokenizer_fn(rec, fldtkn_args):
    column_to_top_values = fldtkn_args[f'column_to_top_values']
    
    d = {}
    for key in column_to_top_values:
        top_values = column_to_top_values[key]
        value = rec.get(key, 'unk')
        if value not in top_values and value != 'unk': value = 'minor'
        key_value = f"{key}_{value}"  # Concatenate key and value
        d[key_value] = 1

    tkn = list(d.keys())
    wgt = list(d.values())
    output = {'tkn': tkn, 'wgt': wgt}
    return output


MetaDict = {
	"column_to_top_values": column_to_top_values,
	"idx2tkn": idx2tkn,
	"tokenizer_fn": tokenizer_fn
}