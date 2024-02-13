
###################################
TaskName = 'EduRxPred'
###################################
PROJECT = 'DrFirst'
# SPACE = {
#     # 'WORKSPACE_PATH': WORKSPACE_PATH, 
#     'DATA_RAW': f'../{PROJECT}-RFT-WorkSpace/0-Data_Raw',
#     'DATA_RFT': f'../{PROJECT}-RFT-WorkSpace/1-Data_RFT',
#     'CODE_FN': f'../{PROJECT}-RFT-WorkSpace', 
#     'CODE_RFT': f'../{PROJECT}-RFT-WorkSpace',
#     'DATA_TASK': f'1-Data_{TaskName}',
#     'MODEL_TASK': f'2-Model_{TaskName}', 
# }

SPACE = {
    # 'WORKSPACE_PATH': WORKSPACE_PATH, 
    'DATA_RAW': f'../_Data/0-Data_Raw',
    'DATA_RFT': f'../_Data/1-Data_RFT',
    'DATA_CaseObs': f'../_Data/2-Data_CaseObs',
    # 'CODE_FN': f'..', 
    'CODE_FN': f'../pipeline', 
    'CODE_RFT': f'../pipeline',
    'DATA_TASK': f'Data_{TaskName}',
    'MODEL_TASK': f'Model_{TaskName}', 
}