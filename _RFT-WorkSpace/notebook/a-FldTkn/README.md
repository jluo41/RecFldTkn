# FldTkn


# 3. Process Field Token (Pipeline B) $\phi$. 

You can develop the record-level function in `/notebook/RecFolder/xxx.ipynb`.

The developed `\phi` functions will be saved in `../fn_fldtkn`.

How to define the `\phi`?


## 3.1 CateTkn

**CateTkn with Limited Token Size**.

*gender*: Gender as an attribute, its value could be 'M', 'F', None. Then its attribute values could be 'Gender-M', 'Gender-F', 'Gender-None'. In this way, this attribute could contribute to 3 feature names (token names). For each of the token, we can assign a weight to it. 
> For example, `{"gender": 'F'}` can be converted into `{'tkn': ['Gender-M', 'Gender-F', 'Gender-None'], 'wgt': [0, 1, 0]}`. By removing the tkn's wgt = 0, this one could be further reduce to `{'tkn': ['Gender-F'], 'wgt': [1]}`

**CateTkn with Large Token Size**.

*rx_bin*: RxBin is an attribute for insurance information. Its attribute values could be sample from 3k insurance name vocabulary. If we convert all of them into feature list, the feature sequence could be very long. So we will reduce the vocabulary size by keeping the top-freq insurance name. 

The following could explain how we do this by using `columns_to_top_values`:

```python
cols = ['rx_bin', 'rx_pcn', 'plan_name']
column_to_top_values = {}
for col in cols:
    # -- keep top 30 
    top_tkn = list(dfHumanRecAttr[col].value_counts().iloc[:30].index) 
    print(col, len(top_tkn), top_tkn)
    column_to_top_values[col] = top_tkn # tolist()

fldtkn_args[f'column_to_top_values'] = column_to_top_values
```

In this case, we can get a token / feature sequence which is sampled from the 30 length vocabulary. 

**CateTkn with Several Attributes**
Sometimes you can put several attributes together based on the same topic to generate the feature sequence. For example, `P-DemoTkn` takes attribute `age`, `gender`, and `Rx-InsCateTkn` takes attributes `['rx_bin', 'rx_pcn', 'plan_name']` to generate the feature sequence. 

## 3.2 NumeTkn

NumeTkn just put the original attribute as the tkn and the attribute value (as a float number) as the wgt. 

For example: `P-Zip3DemoNumeTkn`

```python
{'tkn': [
    'Male', 'Female', 'Above65YearsOld', 'Above18YearsOld', 
    'White', 'BlackPeople', 'Asian', 'Male-median', 
    'Female-median', 'Above65YearsOld-median', 'Above18YearsOld-median', 'White-median', 'BlackPeople-median', 'Asian-median', 
    'Male-var', 'Female-var', 'Above65YearsOld-var', 'Above18YearsOld-var', 'White-var', 'BlackPeople-var', 'Asian-var'
    ], 
'wgt': [
    5949.16, 6003.56, 2006.96, 9138.8, 10227.94, 
    597.46, 203.68, 2880.5, 2876.5, 1099.0, 4450.5, 
    5287.0, 38.0, 12.0, 48001908.3, 49431285.44, 
    5139801.55, 113653040.78, 130114482.59, 
    3669921.52, 252060.3]}
```

Example `Rx-QuantNumeTkn`.

For an Rx record $r_{ik}^{Rx}$ with the following information:
```python
record = {
    'PID': 1000001, 
    'PInvID': '1000001-005', 
    'RxID': '1000001-005-000', 
    'refills_available': 5, 
    'quantity': 2.0, 
    'days_supply': nan}
```

The generated feature $z_{ik}^{Rx-QuantNume}$ is:
```python
{'tkn': ['refills_available', 'quantity', 'days_supply_None'], 'wgt': [5.0, 2.0, 1]}
```

Pay attention here that the attribute `days_supply` is converted to the token/feature `days_supply_nan` as it is a missing value. We will not put missing value in `wgt`. 


## 3.3 N2CTkn

Sometimes we want to convert numeric attributes into categorical features by setting the interval, min, and max. 

Here is an example:

For an Rx record $r_ik^{Rx}$ with the following information:
```python
record = {
    'PID': 1000001, 
    'PInvID': '1000001-005', 
    'RxID': '1000001-005-000', 
    'refills_available': 5, 
    'quantity': 2.0, 
    'days_supply': nan}
```

We predefine the interval settings for them. 
```python
item_to_configs = {
    'refills_available': {'Max': 10, 'Min': 0, 'INTERVAL': 1}, 
    'quantity': {'Max': 200, 'Min': 0, 'INTERVAL': 10}, 
    'days_supply': {'Max': 365, 'Min': 0, 'INTERVAL': 30}, 
}
```

Then we can get the following feature dictionary:

```python
{'tkn':  [
    'refills_available:5~6', 'refills_available:5~6Level', 
    'quantity:0~10', 'quantity:0~10Level', 
    'days_supply:None'], 
'wgt': [
    1, 0.0, 
    1, 0.2, 
    1]}
```

This is just another representation of the record $r$. This feature size will be useful for the deep learning models. 


## 3. 4 TxtTkn

TODO. 




# [Step 0]: SPACE

Updating the SPACE and working environments. 

# Part 1: Get dfRecAttr

## [Step 1]: Load RecAttr

```python
###########################
FldTknName = 'RecName-FldTknXXXXX'# <-------- select your yaml file name
FldType = 'Cate'
###########################
```

Updating the `record.yaml` file. 


## [Step 2]: Updating Yaml File

Updating yaml file for record-field-token. 

```yaml
```


## [Step 3]: Pre-defined Token Vocab

* `Cate`

* `N2C`

* `Nume`

* `External`:


## [Step 4]: Tokenizer

$\phi$: converting a record into a feature matrix. 

## [Step 5]: Vocab


* `Cate`

* `N2C`

* `Nume`

* `External`:


## [Step 6]: Save to pipeline files

## [Step 7]: Application
