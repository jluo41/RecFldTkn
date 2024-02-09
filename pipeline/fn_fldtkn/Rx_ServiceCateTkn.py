import pandas as pd

import numpy as np

idx2tkn = ['send_refill_reminder_messages:nan', 'send_renewal_reminder_messages:nan', 'supports_hippo_prices:nan', 'supports_hippo_prices65:nan', 'supports_copay_prices:nan', 'supports_hippo_prices_medicare:nan', 'send_refill_reminder_messages:True', 'send_renewal_reminder_messages:True', 'supports_hippo_prices:True', 'supports_hippo_prices65:True', 'supports_copay_prices:True', 'supports_hippo_prices_medicare:True', 'send_refill_reminder_messages:False', 'send_renewal_reminder_messages:False', 'supports_hippo_prices:False', 'supports_hippo_prices65:False', 'supports_copay_prices:False', 'supports_hippo_prices_medicare:False', 'send_refill_reminder_messages:None', 'send_renewal_reminder_messages:None', 'supports_hippo_prices:None', 'supports_hippo_prices65:None', 'supports_copay_prices:None', 'supports_hippo_prices_medicare:None']

def tokenizer_fn(rec, fldtkn_args):
    d = {}
    val_cols = ['send_refill_reminder_messages', 'send_renewal_reminder_messages',
       'supports_hippo_prices', 'supports_hippo_prices65',
       'supports_copay_prices', 'supports_hippo_prices_medicare']

    for col in val_cols:

        # colvalue = int(rec.get(col, 'False') == 'True')
        colvalue = rec[col]
        d[col + ':' + str(colvalue)] = 1
        # d[col] = str(int(rec.get(col, 0)))
        
    key = list([k for k in d])
    wgt = list([v for k, v in d.items()])
    output = {'tkn': key, 'wgt': wgt}
    return output


MetaDict = {
	"idx2tkn": idx2tkn,
	"tokenizer_fn": tokenizer_fn
}