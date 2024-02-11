# Repo Structure

This repo is for the data process purpose, and will be used as pipeline to convert the `rawrec` (Raw Record) into the `recfld` (Clean Record Table) and `fldtkn` (Additional Feature set).

* `notebook`: the jupyter notebook which will be used to develop the code


# Proj_Space

```python
###################################
TaskName = 'RFT'
###################################
PROJECT = 'YourProjectGroup' # WellDoc, MedStar ...
SPACE = { 
    'DATA_RAW': f'../_Data/0-Data_Raw',
    'DATA_RFT': f'../_Data/1-Data_RFT',
    'CODE_FN': f'../pipeline', 
    'CODE_RFT':  f'../pipeline', 
    'DATA_TASK': f'../Data/Data_{TaskName}',
    'MODEL_TASK': f'../Model/Model_{TaskName}', 
}
```

