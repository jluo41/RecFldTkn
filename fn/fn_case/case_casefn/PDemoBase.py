import pandas as pd

import numpy as np

CaseFnName = "PDemoBase"

Ckpd_to_CkpdObsConfig = {}

RO_to_ROName = {'RO': 'hP.rP'}

ROName_to_RONameInfo = {'hP.rP': {'HumanName': 'P', 'RecordName': 'P'}}

HumanRecordRecfeat_Args = {'P': {'P': []}}

COVocab = {'idx2tkn': [], 'tkn2tid': {}}

def fn_CaseFn(case_example,     # <--- case to process
               ROName_list,      # <--- from COName
               ROName_to_ROData, # <--- in scope of case_example
               ROName_to_ROInfo, # <--- in scope of CaseFleshingTask
               COVocab,          # <--- in scope of CaseFleshingTask, from ROName_to_ROInfo
               caseset,          # <--- in scope of CaseFleshingTask,
               ):

    assert len(ROName_list) == 1
    ROName = ROName_list[0]

    #############################################
    ROData = ROName_to_ROData[ROName]
    df = ROData# .to_pandas() 
    # display(df)

    rec = df.iloc[0]
    # Define a function to map age to age group
    def map_age_to_group(age):
        """
        Maps an age value to an age group category.

        Args:
            age (int): The age to categorize

        Returns:
            str: Age group category ("0-17", "18-39", "40-64", or "65+")
        """
        if age < 18:
            return "0-17"
        elif age < 40:
            return "18-39"
        elif age < 65:
            return "40-64"
        else:
            return "65+"


    def map_disease_type(disease_type):
        if disease_type == '1.0':
            return 'T1D'
        elif disease_type == '2.0':
            return 'T2D'
        else:
            return disease_type


    def map_gender(gender):
        if gender == 2:
            return 'Male'
        elif gender == 1:
            return 'Female'
        else:
            return gender


    d = {}
    d['Tag-Gender'] = map_gender(rec['Gender'])
    d['Num-Age'] = case_example['ObsDT'].year - rec['YearOfBirth']
    d['Tag-AgeGroup'] = map_age_to_group(d['Num-Age'])
    d['Tag-DiseaseType'] = map_disease_type(rec['DiseaseType'])
    d['Tag-Regimen'] = rec['MRSegmentID']

    return d


MetaDict = {
	"CaseFnName": CaseFnName,
	"Ckpd_to_CkpdObsConfig": Ckpd_to_CkpdObsConfig,
	"RO_to_ROName": RO_to_ROName,
	"ROName_to_RONameInfo": ROName_to_RONameInfo,
	"HumanRecordRecfeat_Args": HumanRecordRecfeat_Args,
	"COVocab": COVocab,
	"fn_CaseFn": fn_CaseFn
}