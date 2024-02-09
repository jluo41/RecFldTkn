import pandas as pd

import numpy as np

column_to_top_values = {'pharmacy_name_rx': ['EXPRESS SCRIPTS', 'HEB PHARMACY', 'CENTERWELL PHARMACY, INC.', 'OPTUMRX, INC.', 'THE MEDICINE SHOPPE PHARMACY', 'KROGER PHARMACY', 'GENOA HEALTHCARE, LLC', 'WINN DIXIE', 'DOD FT LIBERTY PHARMACY', 'GIANT PHARMACY', 'DILLON PHARMACY', 'AHF PHARMACY', 'FRYS FOOD AND DRUG', 'CMC PHARMACY', 'INGLES PHARMACY', 'CAREMARK ILLINOIS MAIL PHARMACY, LLC', 'HERCHE BLOOR PHARMACY', 'KING SOOPERS PHARMACY', 'EXACTCARE', 'PACKARD DISCOUNT PHARMACY', 'STOP & SHOP PHARMACY', 'KINNEY DRUGS', 'PROFESSIONAL PHARMACY OF DARLINGTON', 'BORINQUEN PHARMACY', 'HOMETOWN PHARMACY', 'BROOKSHIRE BROTHERS PHARMACY', 'LEROY PHARMACY #1', 'JONES & COUNTS PHARMACY', 'PBM MEDS BY MAIL CHEYENNE PHARMACY', 'MESQUITE EMPLOYEE HEALTH CENTER PHARMACY'], 'specialty_pharmacy': [0, 1]}

idx2tkn = ['pharmacy_name_rx_unk', 'pharmacy_name_rx_minor', 'pharmacy_name_rx_EXPRESS SCRIPTS', 'pharmacy_name_rx_HEB PHARMACY', 'pharmacy_name_rx_CENTERWELL PHARMACY, INC.', 'pharmacy_name_rx_OPTUMRX, INC.', 'pharmacy_name_rx_THE MEDICINE SHOPPE PHARMACY', 'pharmacy_name_rx_KROGER PHARMACY', 'pharmacy_name_rx_GENOA HEALTHCARE, LLC', 'pharmacy_name_rx_WINN DIXIE', 'pharmacy_name_rx_DOD FT LIBERTY PHARMACY', 'pharmacy_name_rx_GIANT PHARMACY', 'pharmacy_name_rx_DILLON PHARMACY', 'pharmacy_name_rx_AHF PHARMACY', 'pharmacy_name_rx_FRYS FOOD AND DRUG', 'pharmacy_name_rx_CMC PHARMACY', 'pharmacy_name_rx_INGLES PHARMACY', 'pharmacy_name_rx_CAREMARK ILLINOIS MAIL PHARMACY, LLC', 'pharmacy_name_rx_HERCHE BLOOR PHARMACY', 'pharmacy_name_rx_KING SOOPERS PHARMACY', 'pharmacy_name_rx_EXACTCARE', 'pharmacy_name_rx_PACKARD DISCOUNT PHARMACY', 'pharmacy_name_rx_STOP & SHOP PHARMACY', 'pharmacy_name_rx_KINNEY DRUGS', 'pharmacy_name_rx_PROFESSIONAL PHARMACY OF DARLINGTON', 'pharmacy_name_rx_BORINQUEN PHARMACY', 'pharmacy_name_rx_HOMETOWN PHARMACY', 'pharmacy_name_rx_BROOKSHIRE BROTHERS PHARMACY', 'pharmacy_name_rx_LEROY PHARMACY #1', 'pharmacy_name_rx_JONES & COUNTS PHARMACY', 'pharmacy_name_rx_PBM MEDS BY MAIL CHEYENNE PHARMACY', 'pharmacy_name_rx_MESQUITE EMPLOYEE HEALTH CENTER PHARMACY', 'specialty_pharmacy_unk', 'specialty_pharmacy_minor', 'specialty_pharmacy_0', 'specialty_pharmacy_1']

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