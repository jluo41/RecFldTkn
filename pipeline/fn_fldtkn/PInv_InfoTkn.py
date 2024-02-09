import pandas as pd

import numpy as np

column_to_top_values = {'invitation_type': ['PrescribingEvent', 'Followup', 'Refill', 'Renewal', 'Reinforcement'], 'invitation_state': ['Delivered', 'Ignored', 'SMSBlocked', 'Processed', 'New'], 'workflow_step': ['1', '2', 'InitialMessage', 'FollowupMessage', 'ReinforcementMessage']}

item_to_configs = {}

idx2tkn = ['invitation_type_unk', 'invitation_type_minor', 'invitation_type_PrescribingEvent', 'invitation_type_Followup', 'invitation_type_Refill', 'invitation_type_Renewal', 'invitation_type_Reinforcement', 'invitation_state_unk', 'invitation_state_minor', 'invitation_state_Delivered', 'invitation_state_Ignored', 'invitation_state_SMSBlocked', 'invitation_state_Processed', 'invitation_state_New', 'workflow_step_unk', 'workflow_step_minor', 'workflow_step_1', 'workflow_step_2', 'workflow_step_InitialMessage', 'workflow_step_FollowupMessage', 'workflow_step_ReinforcementMessage']

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
	"item_to_configs": item_to_configs,
	"idx2tkn": idx2tkn,
	"tokenizer_fn": tokenizer_fn
}