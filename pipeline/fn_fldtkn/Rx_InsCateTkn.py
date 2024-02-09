import pandas as pd

import numpy as np

column_to_top_values = {'rx_bin': [4336.0, 610097.0, 610014.0, 3858.0, 15581.0, 610502.0, 610011.0, 11552.0, 610494.0, 610239.0, 610279.0, 20115.0, 20099.0, 17010.0, 24251.0, 22659.0, 12833.0, 4915.0, 19595.0, 1553.0, 20107.0, 610591.0, 610455.0, 610602.0, 23880.0, 10579.0, 18902.0, 610593.0, 12312.0, 3585.0], 'rx_pcn': ['ADV', '9999', 'MEDDADV', 'A4', '03200000', 'MCAIDADV', '00670000', 'IRX', 'IS', 'FEPRX', 'MA', 'MEDDAET', 'WG', 'CIMCARE', 'BCTX', 'OHRXPROD', '6334225', 'MEDDPRIME', 'FLBC', 'TNM', 'ASPROD1', 'CTRXMEDD', 'KYPROD1', 'OKA01', 'P303018902', '4343', '1215', '01940000', 'MASSPROD', 'PRX00621'], 'plan_name': ['Department of Defense', 'UHCMR MAPD AND MA/RDS', 'CVS|CAREMARK', 'IRXME BB CDH-Y IRXMDTSTR1', 'AETSS BB CDH-N AETMEDD23', 'E23A1 BB CDH-N HPDMEDD', 'UHCMR PDP', 'EXPRESS SCRIPTS', 'OPTUMRX', 'T14 L4Y 10001', 'AMD23 BB CDH-Y AETMEDD23', 'UNITEDHEALTHCARE DUAL COM', 'Careplus D Humana PDP', 'COMMONWEALTH OF KENTUCKY WELLCARE', 'G1K BB CDH-N BBTMPFEPB', 'UPMC', 'DST PHARMACY SOLUTIONS DIRECT', 'MEDICAL RX', 'Gainwell Ohio Medicaid', 'UHC ACIS ENROLLMENTS', 'BMC RETAIL MEDI', 'HP-OHIO', '2023 Cigna MedD PDP Secure 6T-RTL', 'INDIANA MEDICAID', 'QL TMP Z 30 DAYS RETAIL', 'G1K BB CDH-N BBTMPFEPS', 'BMCHP MA MCD', 'COMMONWEALTH OF KENTUCKY AETNA', 'C23H1 BB CDH-N HPDMEDD', 'INGEN BB CDH-N IRXBAS918']}

idx2tkn = ['rx_bin_unk', 'rx_bin_minor', 'rx_bin_4336.0', 'rx_bin_610097.0', 'rx_bin_610014.0', 'rx_bin_3858.0', 'rx_bin_15581.0', 'rx_bin_610502.0', 'rx_bin_610011.0', 'rx_bin_11552.0', 'rx_bin_610494.0', 'rx_bin_610239.0', 'rx_bin_610279.0', 'rx_bin_20115.0', 'rx_bin_20099.0', 'rx_bin_17010.0', 'rx_bin_24251.0', 'rx_bin_22659.0', 'rx_bin_12833.0', 'rx_bin_4915.0', 'rx_bin_19595.0', 'rx_bin_1553.0', 'rx_bin_20107.0', 'rx_bin_610591.0', 'rx_bin_610455.0', 'rx_bin_610602.0', 'rx_bin_23880.0', 'rx_bin_10579.0', 'rx_bin_18902.0', 'rx_bin_610593.0', 'rx_bin_12312.0', 'rx_bin_3585.0', 'rx_pcn_unk', 'rx_pcn_minor', 'rx_pcn_ADV', 'rx_pcn_9999', 'rx_pcn_MEDDADV', 'rx_pcn_A4', 'rx_pcn_03200000', 'rx_pcn_MCAIDADV', 'rx_pcn_00670000', 'rx_pcn_IRX', 'rx_pcn_IS', 'rx_pcn_FEPRX', 'rx_pcn_MA', 'rx_pcn_MEDDAET', 'rx_pcn_WG', 'rx_pcn_CIMCARE', 'rx_pcn_BCTX', 'rx_pcn_OHRXPROD', 'rx_pcn_6334225', 'rx_pcn_MEDDPRIME', 'rx_pcn_FLBC', 'rx_pcn_TNM', 'rx_pcn_ASPROD1', 'rx_pcn_CTRXMEDD', 'rx_pcn_KYPROD1', 'rx_pcn_OKA01', 'rx_pcn_P303018902', 'rx_pcn_4343', 'rx_pcn_1215', 'rx_pcn_01940000', 'rx_pcn_MASSPROD', 'rx_pcn_PRX00621', 'plan_name_unk', 'plan_name_minor', 'plan_name_Department of Defense', 'plan_name_UHCMR MAPD AND MA/RDS', 'plan_name_CVS|CAREMARK', 'plan_name_IRXME BB CDH-Y IRXMDTSTR1', 'plan_name_AETSS BB CDH-N AETMEDD23', 'plan_name_E23A1 BB CDH-N HPDMEDD', 'plan_name_UHCMR PDP', 'plan_name_EXPRESS SCRIPTS', 'plan_name_OPTUMRX', 'plan_name_T14 L4Y 10001', 'plan_name_AMD23 BB CDH-Y AETMEDD23', 'plan_name_UNITEDHEALTHCARE DUAL COM', 'plan_name_Careplus D Humana PDP', 'plan_name_COMMONWEALTH OF KENTUCKY WELLCARE', 'plan_name_G1K BB CDH-N BBTMPFEPB', 'plan_name_UPMC', 'plan_name_DST PHARMACY SOLUTIONS DIRECT', 'plan_name_MEDICAL RX', 'plan_name_Gainwell Ohio Medicaid', 'plan_name_UHC ACIS ENROLLMENTS', 'plan_name_BMC RETAIL MEDI', 'plan_name_HP-OHIO', 'plan_name_2023 Cigna MedD PDP Secure 6T-RTL', 'plan_name_INDIANA MEDICAID', 'plan_name_QL TMP Z 30 DAYS RETAIL', 'plan_name_G1K BB CDH-N BBTMPFEPS', 'plan_name_BMCHP MA MCD', 'plan_name_COMMONWEALTH OF KENTUCKY AETNA', 'plan_name_C23H1 BB CDH-N HPDMEDD', 'plan_name_INGEN BB CDH-N IRXBAS918']

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