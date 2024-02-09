import pandas as pd

import numpy as np

idx2tkn = ['refills_available', 'quantity', 'days_supply', 'refills_available_None', 'quantity_None', 'days_supply_None']

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