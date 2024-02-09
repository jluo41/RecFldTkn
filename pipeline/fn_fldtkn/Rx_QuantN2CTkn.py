import pandas as pd

import numpy as np

item_to_configs = {'refills_available': {'Max': 10, 'Min': 0, 'INTERVAL': 1}, 'quantity': {'Max': 200, 'Min': 0, 'INTERVAL': 10}, 'days_supply': {'Max': 365, 'Min': 0, 'INTERVAL': 30}}

idx2tkn = ['days_supply:0~30', 'days_supply:0~30Level', 'days_supply:120~150', 'days_supply:120~150Level', 'days_supply:150~180', 'days_supply:150~180Level', 'days_supply:180~210', 'days_supply:180~210Level', 'days_supply:210~240', 'days_supply:210~240Level', 'days_supply:240~270', 'days_supply:240~270Level', 'days_supply:270~300', 'days_supply:270~300Level', 'days_supply:300~330', 'days_supply:300~330Level', 'days_supply:30~60', 'days_supply:30~60Level', 'days_supply:330~360', 'days_supply:330~360Level', 'days_supply:360~390', 'days_supply:360~390Level', 'days_supply:60~90', 'days_supply:60~90Level', 'days_supply:90~120', 'days_supply:90~120Level', 'days_supply:Above365', 'days_supply:Below0', 'days_supply:None', 'quantity:0~10', 'quantity:0~10Level', 'quantity:100~110', 'quantity:100~110Level', 'quantity:10~20', 'quantity:10~20Level', 'quantity:110~120', 'quantity:110~120Level', 'quantity:120~130', 'quantity:120~130Level', 'quantity:130~140', 'quantity:130~140Level', 'quantity:140~150', 'quantity:140~150Level', 'quantity:150~160', 'quantity:150~160Level', 'quantity:160~170', 'quantity:160~170Level', 'quantity:170~180', 'quantity:170~180Level', 'quantity:180~190', 'quantity:180~190Level', 'quantity:190~200', 'quantity:190~200Level', 'quantity:200~210', 'quantity:200~210Level', 'quantity:20~30', 'quantity:20~30Level', 'quantity:30~40', 'quantity:30~40Level', 'quantity:40~50', 'quantity:40~50Level', 'quantity:50~60', 'quantity:50~60Level', 'quantity:60~70', 'quantity:60~70Level', 'quantity:70~80', 'quantity:70~80Level', 'quantity:80~90', 'quantity:80~90Level', 'quantity:90~100', 'quantity:90~100Level', 'quantity:Above200', 'quantity:Below0', 'quantity:None', 'refills_available:0~1', 'refills_available:0~1Level', 'refills_available:10~11', 'refills_available:10~11Level', 'refills_available:1~2', 'refills_available:1~2Level', 'refills_available:2~3', 'refills_available:2~3Level', 'refills_available:3~4', 'refills_available:3~4Level', 'refills_available:4~5', 'refills_available:4~5Level', 'refills_available:5~6', 'refills_available:5~6Level', 'refills_available:6~7', 'refills_available:6~7Level', 'refills_available:7~8', 'refills_available:7~8Level', 'refills_available:8~9', 'refills_available:8~9Level', 'refills_available:9~10', 'refills_available:9~10Level', 'refills_available:Above10', 'refills_available:Below0', 'refills_available:None']

def tokenizer_fn(rec, fldtkn_args):
    
    item_to_configs = fldtkn_args['item_to_configs']
    d = {}
    for item, configs in item_to_configs.items():
        Max = configs['Max']
        Min = configs['Min']
        INTERVAL = configs['INTERVAL']
        if pd.isnull(rec.get(item, None)):
            d[f"{item}:None"] = 1
        elif float(rec[item]) > Max:
            d[ f"{item}:Above{Max}"] = 1
        elif float(rec[item]) < Min:
            d[ f"{item}:Below{Min}"] = 1
        else:
            lower_bound = int((float(rec[item]) // INTERVAL) * INTERVAL)
            upper_bound = int(lower_bound + INTERVAL)
            # Calculate the proportion of BGValue within the interval
            proportion = (float(rec[item]) - lower_bound) / INTERVAL
            # Construct the keys
            key1 = f"{item}:{lower_bound}~{upper_bound}"
            key2 = f"{key1}Level"
            # Add them to the dictionary with appropriate weights
            d[key1] = 1
            d[key2] = proportion

    # Convert dictionary to the desired format
    key = [k for k, w in d.items()]
    wgt = [round(w, 2) for k, w in d.items()]
    output = {'tkn': key, 'wgt': wgt}
    return output


MetaDict = {
	"item_to_configs": item_to_configs,
	"idx2tkn": idx2tkn,
	"tokenizer_fn": tokenizer_fn
}