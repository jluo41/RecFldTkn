import pandas as pd

import numpy as np

idx2tkn = ['top_200_branded_drugs', 'top_50_generic_drugs', 'top_200_branded_drugs_None', 'top_50_generic_drugs_None']

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