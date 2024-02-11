# Human2RecNum


# [Step 1]: Cohort: Pick A Cohort

Adding `cohort_name` and `cohort_label`. 

With the `args_information`, adding the cohort_label.

```python
args_information = ['--cohort_label', '1']
```

# [Step 2]: Helper Function

* `selected_source_file_suffix_list`

* `excluded_cols`: TODO explain this variable names. 
When filter the Human based on the Record Name. 

We want some patients with at least some record numbers. 

* `get_id_column`. 
Input `columns `: the columns of the raw record table.
Get the raw patient id columns. 

* `get_tablename_from_file`. 
Converting file_path to a record name.

# [Step 3]: Process dfHumanRec

```python
df_Human = get_cohort_level_record_number_counts(cohort_name, cohort_label, cohort_args, filepath_to_rawdf)
```

`OneCohort_config`: One Cohort Information: OneCohort_config

`FolderPath`: OneCohort_config['FolderPath']. The folder name for the cohort. 

`filepath_to_rawdf`: raw file path. 


### Section 1: Design and Load the cohort information

- excluded_cols
- selected_source_file_suffix_list
- get_id_column
- get_tablename_from_file

### Section 2: Update filepath_to_rawdf

- filepath_to_rawdf: can be a dictionary. 
This is a dictionary.
The key is the full path csv file. 

### Section 3: Collect Results from the Raw Data. 

For a list of `file_path` in filepath_to_rawdf. 


### Section 4: Filtering the df_Human who does not have any records. 

`excluded_cols`: actually the excluded raw tables. 

### Section 5: Adding and Updating RootID. 

Using the PID. 






