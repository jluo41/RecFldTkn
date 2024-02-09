import pandas as pd

import numpy as np

idx2tkn = ['show_rems_campaigns:True', 'show_coupon_campaigns:True', 'show_educational_campaigns:True', 'show_internal_campaigns:True', 'show_target_campaigns:True', 'show_experimental_campaigns:True', 'show_rems_campaigns:False', 'show_coupon_campaigns:False', 'show_educational_campaigns:False', 'show_internal_campaigns:False', 'show_target_campaigns:False', 'show_experimental_campaigns:False', 'show_rems_campaigns:None', 'show_coupon_campaigns:None', 'show_educational_campaigns:None', 'show_internal_campaigns:None', 'show_target_campaigns:None', 'show_experimental_campaigns:None']

def tokenizer_fn(rec, fldtkn_args):
    d = {}
    val_cols = [
       'show_rems_campaigns', 'show_coupon_campaigns',
       'show_educational_campaigns', 'show_internal_campaigns',
       'show_target_campaigns', 'show_experimental_campaigns',
    ]
    for col in val_cols:
        colvalue = rec[col]
        d[col + ':' + str(colvalue)] = 1
        
    key = list([k for k in d])
    wgt = list([v for k, v in d.items()])
    output = {'tkn': key, 'wgt': wgt}
    return output


MetaDict = {
	"idx2tkn": idx2tkn,
	"tokenizer_fn": tokenizer_fn
}