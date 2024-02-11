# RecName

# [Step 0]: SPACE

Updating the SPACE and working environments. 

# Part 1: Get Record Raw

## [Step 1]: Args

Just updating the record name

```python
###########################
RecName = 'TODO'# <-------- select your yaml file name
###########################
```

Creating and updating the `record.yaml` file. 

## [Step 2]: Updating Yaml File

Copy from the record yaml template. 

Updating the yaml files. 

```yaml
```


## [Step 3]: Select a Cohort

Get `cohort_label` and `cohort_name`. 


## [Step 4]: get df_Human and df_Prt

## [Step 5]: build OneCohortRec_args

## [Step 6]: Update Yaml Again

## [Step 7]: HumanRecRaw

# Part 2: Update RawRec to RecAttr

## [Step 1]: HumanRecFld -- Develop RawRec to RecAttr Code

```python
# -. filter out the records we don't need (optional) 

# -. create a new column for raw record id (optional)

# -. have a check that the raw record id is unique

# -. update datetime columns
   
# -. select a DT. TODO: you might need to localize the datetime to local timezone. 

# -. merge with the parent record (a must except Human Records)

# -. sort the table by Parent IDs and DT

# -. create a new column for RecID
```

## [Step 2]: Pin Down Attr Cols

## [Step 3]: RawRec_to_RecAttr_fn

## [Step 4]: Save and Test. 