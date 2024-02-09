import pandas as pd

import numpy as np

idx2tkn = ['ther_eq_ult_child_ind', 'ther_eq_ult_parent_etc_id', 'ther_eq_hierarchy_level', 'therapeutic_equivalence_id', 'ther_eq_ult_child_ind_None', 'ther_eq_ult_parent_etc_id_None', 'ther_eq_hierarchy_level_None', 'therapeutic_equivalence_id_None']

def tokenizer_fn(rec, fldtkn_args):
    d = {}
    for col in fldtkn_args['value_cols']:
        x = rec[col]
        if pd.isnull(x):
            d[f'{col}_None'] = 1
        else:
            d[col] = float(x)
    # Convert dictionary to the desired format
    key = [k for k, w in d.items()]
    wgt = [round(w, 2) for k, w in d.items()]
    output = {'tkn': key, 'wgt': wgt}
    return output


MetaDict = {
	"idx2tkn": idx2tkn,
	"tokenizer_fn": tokenizer_fn
}