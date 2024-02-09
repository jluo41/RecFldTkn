import pandas as pd

import numpy as np

BATCHED = "1"

def get_case_op_vocab(co_to_CaseObsVocab):

    CaseTaskOpVocab = {}

    co_Y_list = [i for i in co_to_CaseObsVocab if 'TknY' in i]
    co_X_list = [i for i in co_to_CaseObsVocab if 'TknY' not in i]
    
    sequence_name = 'input_ids'
    idx2tkn_all = []
    for co in co_X_list:
        CaseObsVocab = co_to_CaseObsVocab[co] 
        idx2tkn = [co+':'+ tkn for tid, tkn in CaseObsVocab['tkn']['tid2tkn'].items()]
        idx2tkn_all = idx2tkn_all + idx2tkn
    tid2tkn_all = {i: tkn for i, tkn in enumerate(idx2tkn_all)}
    tkn2tid_all = {tkn: i for i, tkn in enumerate(idx2tkn_all)}
    CaseTaskOpVocab[sequence_name] = {'tid2tkn': tid2tkn_all, 'tkn2tid': tkn2tid_all}
    
    if len(co_Y_list) == 1:
        sequence_name = 'labels'
        co_Y = co_Y_list[0]
        co = co_Y
        CaseObsVocab = co_to_CaseObsVocab[co]
        idx2tkn_all = [co+':'+ tkn for tid, tkn in CaseObsVocab['tkn']['tid2tkn'].items()]
        tid2tkn_all = {i: tkn for i, tkn in enumerate(idx2tkn_all)}
        tkn2tid_all = {tkn: i for i, tkn in enumerate(idx2tkn_all)}
        CaseTaskOpVocab[sequence_name] = {'tid2tkn': tid2tkn_all, 'tkn2tid': tkn2tid_all}
    
    return CaseTaskOpVocab


def fn_Case_Operation_Tkn(examples, co_to_CaseObsVocab, CaseTaskOpVocab):
    num_examples = len(examples[[i for i in examples.keys()][0]])
    
    idx_to_examples = {i: {k: v[i] for k, v in examples.items()} for i in range(num_examples)}

    co_Y_list = [i for i in co_to_CaseObsVocab if 'TknY' in i]
    co_X_list = [i for i in co_to_CaseObsVocab if 'TknY' not in i]
    # print(co_Y)
    # print(co_X_list)
    
    if len(co_Y_list) == 1:
        results = {'input_ids': [], 'input_wgts': [], 'labels': [], 'label_wgts': []}
    else:
        results = {'input_ids': [], 'input_wgts': []}

    for idx, example in idx_to_examples.items():
        result_case = {}
        # ============================================ # 
        # get input_ids and input_wgts
        sequence_name = 'input_ids' 
        tkn2tid = CaseTaskOpVocab[sequence_name]['tkn2tid']

        X_tid_updated_total = []
        X_wgt_total = []
        for co in co_X_list:
            CaseObsVocab = co_to_CaseObsVocab[co]
            X_tid = example[co +'_tid']
            X_tkn = [co + ':' + CaseObsVocab['tkn']['tid2tkn'][tid] for tid in X_tid]
            X_tid_updated = [tkn2tid[i] for i in X_tkn]
            X_wgt = example[co +'_wgt']
            X_tid_updated_total = X_tid_updated_total + X_tid_updated
            X_wgt_total = X_wgt_total + X_wgt
        
        # version 1: for DL and ML
        result_case['input_ids'] = X_tid_updated_total
        result_case['input_wgts'] = X_wgt_total

        # version 2: for ML
        # tid2wgt = dict(zip(X_tid_updated_total, X_wgt_total))
        # whole_ids = list(range(len(tkn2tid)))
        # result_case['input_ids'] = whole_ids
        # result_case['input_wgts'] = [tid2wgt.get(tid, 0) for tid in whole_ids]

        # ============================================ # 
        if len(co_Y_list) == 1:
            co_Y = co_Y_list[0]
            # get labels and label_wgts
            sequence_name = 'labels'
            tkn2tid = CaseTaskOpVocab[sequence_name]['tkn2tid']
            CaseObsVocab = co_to_CaseObsVocab[co_Y]
            Y_tid = example[co_Y+'_tid']
            Y_tkn = [co_Y + ':' + CaseObsVocab['tkn']['tid2tkn'][tid] for tid in Y_tid]
            Y_tid_updated = [CaseTaskOpVocab[sequence_name]['tkn2tid'][i] for i in Y_tkn]
            Y_wgt = example[co_Y+'_wgt']

            # version 1: for DL and ML
            result_case['labels'] = Y_tid_updated
            result_case['label_wgts'] = Y_wgt

            # version 2: for ML (consume too much memory and time)
            # tid2wgt = dict(zip(Y_tid_updated, Y_wgt))
            # whole_ids = list(range(len(tkn2tid)))
            # result_case['labels'] = whole_ids
            # result_case['label_wgts'] = [tid2wgt.get(tid, 0) for tid in whole_ids]

        for k, v in result_case.items(): results[k].append(v)
        
    return results 


MetaDict = {
	"BATCHED": BATCHED,
	"get_case_op_vocab": get_case_op_vocab,
	"fn_Case_Operation_Tkn": fn_Case_Operation_Tkn
}