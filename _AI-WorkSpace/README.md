

# EduRxPred and ML Modeling

# Repo Structure

## a. `run_case_scope_whole`

This python script takes the patient $p_i$'s CGM record set $R_i^{cgm}$ as the trigger records, and uses them to the case set $C_i^{scope}$. 

> Suppose a patient have 100 Rx record, and 70 of them shown the Education Button, then we have 70 cases $c_{ij} = (p_i, t_{ij})$ from $C_i^{scope}$, where $p_i$ is the patient and $t_{ij}$ is the observation time.


At the same time, for the patient $p_i$, the script `run_case_scope_whole.py` will classify him/her into a group based on $p_i$'s `age_bucket`, `gender`. 


**Usage**:
```shell
python run_case_scope_whole_EduRxPred.py \
    --record_name "Rx" \
    --case_id_columns "PID" "ObsDT" "PInvID" "RxID" \
    --special_columns "show_educational_campaigns"
```

You can find the scope case $C^{scope}$ for all patients in the folder: 
 `Data_EduRxPred/CaseFolder/{groupid}_{groupname}_whole.p`. 

**Notebook**:

The notebook `notebook/a-run_case_scope_whole.ipynb` is the notebook to develop this script. 


## b. `run_case_observations`

### b.1 description

The python script `run_case_observations.py` get the case level observation to a case $c_{ij} = (p_i, t_{ij})$. 


In mathematics, we have a case-level feature to calculate, and call it `CaseToken`. Then it can be noted as case level feature function $\Phi^{casetkn}$ , that $a_{ij} = \Phi^{casetkn}(c_{ij}, R_i)$. 

* $c_{ij}$ is the case of $(p_i, t_{ij})$
* $R_{i}$ is the patient $p_i$'s record set. $R_{i} = \cup R_i^{recname}$, where $recname$ is the name for different record types.
* $\Phi^{casetkn}$ is the function to get the case level features (token list) $a_{ij}$ to case $c_{ij}$ at the observation time $t_{ij}$, based on the patient $p_i$'s record set $R_i$. 

### b.2 Types of $o_{ij}$ and $\Phi$
Only subsets of $R_{i}$ will be used to calculate $a_{ij}$. There are different types of $a_{ij}$ and $\Phi^{casetkn}$ based on the subset of $R_i$ used to calculate the features.
* Standing at the observation time of $t_{ij}$ for case $c_{ij}$, the records happened before $t_{ij}$ is the before-record set: $R_i^{bf}$, and records happened after $t_{ij}$ is the after-record set: $R_i^{af}$. 

* If $R_i^{bf}$ is the input to $\Phi^{casetkn}$, the returned feature $a_{ij}$ will be used as the input features $x_{ij}$.
* If $R_i^{af}$ is the input to $\Phi^{casetkn}$, the returned feature $a_{ij}$ will be used as the future outcome label $y_{ij}$. 

* Only the case with both $x_{ij}$ and $y_{ij}$ can because an AI model development point: $(x_{ij}, y_{ij}) \in C_i^{dev}$. 

* We use the `CheckPeriod` $ckpd_{ij}$ anchored in $t_{ij}$ to select $R_i^{ckdp_{ij}}$ from $R_i$. The $ckpd$ can be `Bf2M`, `Bf24H`, `Af2H`, etc.

### b.3 Record Observations (Ckpd, RecName, Field)

To get a case observation $a_{ij}$, we need to have one (or many) record observations $recobs$ and a `casetkn` function $\Phi_{casetkn}$.  

To prepare inputs to $\Phi$ at the $c_{ij}$'s observation time $t_{ij}$, we can get an observation of CheckPeriod-Record-Field $recobs$: $(name, ckpd, \phi^{name-fld})$ from $R_i$, where $\phi^{name-fld}$ is record-level feature function. 

* `CheckPeriod`: The check period $ckpd_{ij}$ anchored with observation time $t_{ij}$ in the case $c_{ij}$. The options can be `Bf24H`, `Af2H` etc. 

* `RecName`: $name$ for $R_{i}^{name}$, like `CGM5Min`, `FoodRec` (in the future). Together with `CheckPeriod`, we have $R_{i}^{ckpd_{ij}, name}$.

* `Field` (Optional): The record-level feature function $\phi_{name, fld}$ for the field $fld$. We have $z_k = \phi_{name, fld}(r_k)$, where $r_{k} \in R_{i}^{ckpd_{ij}, name}$. Then we have a record observation: $recobs_i = R_{i}^{ckpd_{ij}, name, \phi^{name-fld}}$

For one case-level function $\Phi$, its inputs can be multiple observation tuples $(name, ckpd, \phi_{name, fld})$. These observations will be processed to case-level features $a_{ij}$. 



### b. 4 Case Observation $\Phi$

**case_tkn**

There are different types of $\Phi$. For each $\Phi$, we will write the funtion tools and save them into the module `fn_casetkn`. For example: `1TknIn5Min`, `RecNum`, etc.

A casetkn is `ro.RecObs1&RecObs2&RecObs3-ct.CaseTkn`

The Case Observation will be save in `xxx`

### b.5 Code

**Usage:**
```shell

python ../run_case_observations.py \
    --group_id 0 \
    --case_type "whole" \
    --case_id_columns "PID" "ObsDT" "PInvID" "RxID" \
    --record_observations 'Rx-InObs' \
    --case_tkn "RecNum" \
    --batch_size 500


python ../run_case_observations.py \
    --group_id 10 \
    --case_type "whole" \
    --case_id_columns "PID" "ObsDT" "PInvID" "RxID" \
    --record_observations 'Rx-InObs-CmpCate' \
    --case_tkn "InCaseTkn" \
    --batch_size 500

python ../run_case_observations.py \
    --group_id 0 \
    --case_type "whole" \
    --case_id_columns "PID" "ObsDT" "PInvID" "RxID" \
    --record_observations 'P-Demo' \
    --case_tkn "InCaseTkn" \
    --batch_size 500


python ../run_case_observations.py \
    --group_id 1 \
    --case_type "whole" \
    --case_id_columns "PID" "ObsDT" "PInvID" "RxID" \
    --record_observations 'EgmEdu-Af1W' \
    --case_tkn "FutEduTkn" \
    --batch_size 500
```


`--group_id`: the id for a group of patients.

`--case_type`: the scope cases type. `whole` means all scope cases. We will have different versions of scope case, i.e., `whole`, `use`, `label`, `dev`, etc. 

`--record_observations`: the observation of RecCkpd $(RecName, Ckpd)$. It can be `EgmEdu-Af1W`, or `Rx-InObs-CmpCate`.

`--case_tkn`: the casetkn function $\Phi$. Here the function $\Phi$ is `RecNum`, which return the record number of the observation $(RecName, Ckpd)$.

`--batch_size`: the number of cases to process each time. 


**Vocab**:

Tokenizer will also be generated based on $\phi^{rec-field}$ and $\Phi^{casetkn}$. 


**Notebook**:

The notebook `notebook/b-run_case_observation.ipynb` is the notebook to develop this script. 


