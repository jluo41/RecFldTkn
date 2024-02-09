import pandas as pd

import numpy as np

idx2tkn = ['brand_source_BRAND', 'brand_source_GENERIC', 'brand_source_unk']

def tokenizer_fn(rec, fldtkn_args):
    val_cols = ['brand_source']
    d = {}
    for col in val_cols:
        value = rec.get(col, 'unk')
        if type(value) != str:
            value = 'unk'
        key_value = f"{col}_{value}"  # Concatenate key and value
        d[key_value] = 1

    tkn = list(d.keys())
    wgt = list(d.values())
    output = {'tkn': tkn, 'wgt': wgt}
    return output


MetaDict = {
	"idx2tkn": idx2tkn,
	"tokenizer_fn": tokenizer_fn
}