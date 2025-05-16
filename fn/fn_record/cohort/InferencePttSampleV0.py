import os

import json

import pandas as pd

import numpy as np

OneCohort_Args = {'CohortLabel': 9,
 'CohortName': '20241013_InferencePttSampleV0',
 'FolderPath': './_Data/0-Data_Raw/Inference/',
 'SourcePath': './_Data/0-Data_Raw/Inference/patient_sample',
 'Source2CohortName': 'InferencePttSampleV0'}

SourceFile_SuffixList = ['json']

def fill_missing_keys(input_form, template):
    # Create a copy of the template to preserve the original structure
    filled_form = template.copy()
    
    # Function to recursively fill missing keys based on the template
    def recurse_fill(current_form, current_template):
        if isinstance(current_template, dict):
            # Ensure current_form is a dictionary
            if not isinstance(current_form, dict):
                current_form = {}
            for key in current_template:
                if isinstance(current_template[key], dict):
                    # If the template value is a dictionary, recurse into it
                    current_form[key] = recurse_fill(current_form.get(key, {}), current_template[key])
                elif isinstance(current_template[key], list) and current_template[key] and isinstance(current_template[key][0], dict):
                    # If the template value is a list of dictionaries, process each dictionary
                    if key in current_form and isinstance(current_form[key], list):
                        # Ensure each element in the input list conforms to the template
                        current_form[key] = [recurse_fill(elem, current_template[key][0]) for elem in current_form[key]]
                    else:
                        # If the key is missing or not a list in the input, initialize it with a list containing a filled template dict
                        current_form[key] = [recurse_fill({}, current_template[key][0])]
                else:
                    # Set the key to None if it is not present in the input and not a dict or list of dicts
                    current_form[key] = current_form.get(key, None)
            return current_form
        elif isinstance(current_template, list) and current_template and isinstance(current_template[0], dict):
            # If the top-level template itself is a list of dictionaries
            if isinstance(current_form, list):
                return [recurse_fill(elem, current_template[0]) for elem in current_form]
            else:
                return [recurse_fill({}, current_template[0])]
        else:
            # Return None for unexpected types or if the template does not contain a dict or list of dicts
            return None

    # Call the recursive function starting with the entire form
    filled_form = recurse_fill(input_form, filled_form)
    return filled_form


def replace_none_with_list(d):
    for key, value in d.items():
        if value is None:
            d[key] = []
        elif isinstance(value, dict):
            replace_none_with_list(value)
    return d


def get_RawName_from_SourceFile(file_path, OneCohort_Args):
    RawName = os.path.basename(file_path).split('.')[0].replace('inference_form_', '')
    return RawName


def get_InferenceEntry(OneCohort_Args, 
                       SourceFile_List,
                       get_RawName_from_SourceFile):
    Inference_EntryPath = {}
    for file_path in SourceFile_List:
        RawName = get_RawName_from_SourceFile(file_path, OneCohort_Args)
        Inference_EntryPath[RawName] = file_path

    Inference_Entry = {}
    RawName = 'template'
    inference_form_path = Inference_EntryPath[RawName]
    with open(inference_form_path, 'r') as f:
        template_form = json.load(f)
    Inference_Entry['template_form'] = template_form

    RawName_list = [i for i in Inference_EntryPath.keys() if 'sample' in i]
    for RawName in RawName_list:
        inference_form_path = Inference_EntryPath[RawName]
        with open(inference_form_path, 'r') as f:
            inference_form = json.load(f)

        # inference_form = fill_missing_keys(inference_form, template_form)
        # inference_form = replace_none_with_list(inference_form)

        Inference_Entry[f'inference_form_{RawName}'] = inference_form

    return Inference_Entry


def process_Source_to_Raw(OneCohort_Args, 
                          SourceFileList_or_InferenceEntry, 
                          get_RawName_from_SourceFile,
                          SPACE):

    # 1. prepare inference_form
    if type(SourceFileList_or_InferenceEntry) == list:
        SourceFile_List = SourceFileList_or_InferenceEntry
        Inference_Entry = get_InferenceEntry(OneCohort_Args, 
                                             SourceFile_List, 
                                             get_RawName_from_SourceFile)
    else:
        Inference_Entry = SourceFileList_or_InferenceEntry

    assert 'template_form' in Inference_Entry
    template_form = Inference_Entry['template_form']
    # print([i for i in template_form])

    inference_form_name_list = [i for i in Inference_Entry if 'inference_form' in i]
    RawName_to_dfRawList = {}
    for inference_form_name in inference_form_name_list:
        inference_form = Inference_Entry[inference_form_name]
        inference_form = fill_missing_keys(inference_form, template_form)
        inference_form = replace_none_with_list(inference_form)

        for RawName in template_form: # <---- pay attention here, we use keys in template_form.
            data = inference_form[RawName]
            # data: {table_name: value_list or []}
            ############################################
            data_new = {}
            max_num = max([0] + [len(v) for v in data.values()])
            for k, v in data.items():
                if len(v) == 0:
                    data_new[k] = [None] * max_num
                else:
                    data_new[k] = v
            for k, v in data_new.items(): assert len(v) == max_num
            df = pd.DataFrame(data_new)
            ############################################


            if RawName not in RawName_to_dfRawList:
                RawName_to_dfRawList[RawName] = []
            RawName_to_dfRawList[RawName].append(df)

    RawName_to_dfRaw = {}
    for RawName, df_list in RawName_to_dfRawList.items():
        df = pd.concat(df_list)
        RawName_to_dfRaw[RawName] = df


    print([i for i in RawName_to_dfRaw])


    ############# 
    df_Patient = RawName_to_dfRaw['Patient']

    try:
        df_UserDetail = RawName_to_dfRaw['UserDetail']
        df_Ptt = pd.merge(df_Patient, df_UserDetail, on='PatientID', how='outer')
    except:
        df_Ptt = df_Patient
    RawName_to_dfRaw['Ptt'] = df_Ptt


    ############################## Diet Information #################################
    # import pandas as pd
    # pd.set_option('display.max_columns', None)

    # Part 1:
    # There is a food items table. Several food items are tied to one CarbEntry. 
    # So we need to aggregate the food items by CarbEntry. 
    ID_to_Type = {
        1:'BeforeBreakfast', 
        2:'AfterBreakfast', 
        3:'BeforeLunch', 
        4:'AfterLunch', 
        5:'BeforeDinner', 
        6:'AfterDinner', 
        7:'Bedtime',
        8: 'BeforeExercise',
        9: 'AfterExercise',
        12: 'Snack',
        14: 'Fasting', 
    }
    # df_food_path = [i for i in SourceFile_List if 'ELogFoodItem' in i][0]
    df_food = RawName_to_dfRaw['ELogFoodItem']
    df_food['ActivityType'] = df_food['ActivityTypeID'].map(ID_to_Type)
    print(df_food.shape, '<-- df_food.shape')

    columns = [
    # 'ELogFoodItemID', 
    'PatientID', 
    'CarbEntryID', 
    'FoodName',
    'EntrySourceID', 
    # 'ActivityTypeID', 
    'ObservationDateTime',
    'ObservationEntryDateTime', 
    'CreatedDateTime', 'ModifiedDateTime', # 'RowVersionID',
    'ObservationStatus', 'ObservationCreatedBy',
    'TimezoneOffset', 'Timezone', 
    'FoodID', 'ServingSize', 'ServingType', 'Carbs', 'Fiber', 'Fat', 'Calories',
    'Protein', 'Sodium', 'ServingsConsumed', #  'ExternalSourceID', 'ExternalEntryID',
    'FoodImageID', 'SaturatedFat', 'PolyUnSaturatedFat',
    'MonoUnSaturatedFat', 'TransFat', 'Cholesterol', 'Potassium', 'Sugar',
    'AddedSugars', 'ActivityType']

    df_food = df_food[columns]
    print(df_food['CarbEntryID'].nunique(), '<-- df_food carbentryid')


    # df_carbs_path = [i for i in SourceFile_List if 'ELogCarbsEntry' in i][0]
    # df_carbs = pd.read_csv(df_carbs_path, low_memory=False)

    df_carbs = RawName_to_dfRaw['ELogCarbsEntry']
    print(df_carbs.shape, '<-- df_carbs.shape')
    # print(df_carbs.shape)

    columns = [
    # 'CarbsEntryID', 
    'PatientID', 'EntryID', 
    # 'EntrySourceID',
    'ActivityTypeID', 
    'ObservationDateTime', 'ObservationEntryDateTime',
    'EntryCreatedDateTime',  'ModifiedDateTime', # 'ExternalSourceID',
    'TimezoneOffset', 'Timezone', 
    # 'ObservationCreatedBy', 'ObservationStatus', # 'RowVersionID',
    # 'SourceReferenceID', 
    'CarbsValue', 
    # 'ExternalEntryID'
    ]

    df_carbs = df_carbs.reindex(columns=columns)
    print(df_carbs['ActivityTypeID'].value_counts()) #  = df_carbs['ActivityTypeID'].astype(str)
    df_carbs['ActivityType'] = df_carbs['ActivityTypeID'].map(ID_to_Type)
    df_carbs['ActivityType'].value_counts()
    del df_carbs['ActivityTypeID']
    df_carbs = df_carbs.rename(columns = {'EntryID': 'CarbEntryID'})
    print(df_carbs.shape, '<-- df_carbs.shape')
    print(df_carbs['CarbEntryID'].nunique(), '<-- df_carbs carbentryid')

    df_food_by_meal = df_food.groupby(['PatientID', 'CarbEntryID']).agg({
        "FoodName": lambda x: "; ".join(x),
        "Carbs": "sum",
        # "ServingsConsumed": "sum",
        # 'ServingSize', 'ServingType',
        "Carbs": "sum",
        "Fiber": "sum",
        "Fat": "sum",
        "Calories": "sum",
        "Protein": "sum",
        "Sodium": "sum",
        "SaturatedFat": "sum",
        "PolyUnSaturatedFat": "sum",
        "MonoUnSaturatedFat": "sum",
        "TransFat": "sum",
        "Cholesterol": "sum",
        "Potassium": "sum",
        "Sugar": "sum",
        "AddedSugars": "sum",
        # "ActivityType": "first",
    })


    df_food_by_meal = df_food_by_meal.reset_index()
    print(df_food_by_meal.shape, '<-- df_food_by_meal.shape')
    # df_food_by_meal
    df_diet = pd.merge(df_carbs, df_food_by_meal, on=['PatientID', 'CarbEntryID'])
    print(df_diet.shape, '<-- df_diet.shape', 'the total number of diet records')
    # df_diet.shape 
    # df_diet.head()
    df_food_by_meal = df_food_by_meal.reset_index()
    df_meal = pd.merge(df_carbs, df_food_by_meal, on=['PatientID', 'CarbEntryID'])

    # file = os.path.join(OneCohort_Args['FolderPath'], 'processed_RawFile_Diet.csv')
    # df_meal.to_csv(file, index=False)
    # RawName_to_dfRaw['Diet'] = file
    RawName_to_dfRaw['Diet'] = df_meal






    ############################ add the medication ##############################
    # import pandas as pd
    # pd.set_option('display.max_columns', None)

    # medadmin_path = [i for i in SourceFile_List if 'MedAdmin' in i][0]
    # print(medadmin_path)
    # df_med = pd.read_csv(medadmin_path, low_memory=False)
    # if 'MedAdmin' in RawName_to_dfRaw:
    df_med = RawName_to_dfRaw.get('MedAdmin', pd.DataFrame())
    columns = [
    'PatientID', 'MedAdministrationID', 'AdministrationID', 'ELogEntryID',
    ###### time
    'AdministrationDate', 
    'UserAdministrationDate', 
    'EntryDateTime',  'CreatedDate', 'ModifiedDateTime', 'AdministrationTimeZoneOffset', 'AdministrationTimeZone',
    ###### 


    ######
    'MedicationID', 'Dose', 
    ######

    'MedSourceID', 
    'AdministrationTimeLabelID',
    'ActivityTypeID',
    # 'StatusID', 
    # 'CreatedBy', # 'RowVersionID', 
    # 'MedPrescriptionID', 
    # 'PrescriptionGUID', 
    # 'MedPrescriptionTime', 
    # 'AdminSlot', 'ScheduledSlot', 

    # 'BGValue', 'CarbsValue',
    # 'InsulinCalculatorUsageStatus', 
    # 'IOBValue', 'FoodInsulinDose',
    # 'ExternalEntryID'
    ]
    # df_med = df_med[columns].reset_index()
    df_med = df_med.reindex(columns=columns)

    # file = os.path.join(OneCohort_Args['FolderPath'], 'processed_RawFile_MedAdmin.csv')
    # df_med.to_csv(file, index=False)
    # RawName_to_dfRaw['MedAdmin'] = file
    RawName_to_dfRaw['MedAdmin'] = df_med





    ############################# add the exercise #############################
    # exercise_path = [i for i in SourceFile_List if 'ELogExercise' in i][0]
    # print(exercise_path)
    df_exercise = RawName_to_dfRaw['ELogExerciseEntry']
    columns = df_exercise.columns
    df_exercise['ActivityType'] = df_exercise['ActivityTypeID'].map(ID_to_Type)
    print(df_exercise.shape)

    columns = [
    'PatientID', 
    # 'EntryID', 
    # 'ExerciseEntryID', 
    'ObservationDateTime', 'ObservationEntryDateTime', 
    # 'EntryCreatedDateTime',  'ModifiedDateTime', #  'ObservationCreatedBy',
    'TimezoneOffset', 'Timezone',
    'ExerciseType', 'ExerciseIntensity', 'TimeSinceExercise',
    'ActivityTypeID', 
    'ExerciseDuration',
    # 'ObservationStatus',
    # 'RowVersionID', 
    # 'SourceReferenceID',
    'CaloriesBurned', 
    'DistanceInMeters', # 'ExternalEntryID',
    # 'ExternalSourceID', # 'HealthConnectMetaDataId'
    # 'EntrySourceID', 
    ]

    # df_exercise = df_exercise[columns]
    df_exercise = df_exercise.reindex(columns=columns)
    id_to_intensity = {
        0: None, 
        1: 'High', 
        2: 'Moderate', 
        3: 'Low', 
    }
    df_exercise['ExerciseIntensity'] = df_exercise['ExerciseIntensity'].map(id_to_intensity)

    id_to_exercise_type = {  
        100: 'Walking',
        101: 'Running',
        102: 'Hiking',
        103: 'Bicycling',
        104: 'Swimming',
        105: 'Strength_training',
        106: 'Home_activities',
        107: 'Gardening__Lawn',
        108: 'Dancing__Aerobics',
        109: 'Skiing__Skating',
        110: 'Yoga_Pilates',
        111: 'Other',
        1: 'Cardiovascular',
        2: 'StrengthTraining',
        3: 'Sports',
        4: 'FitnessClass',
        5: 'YogaPilates',
    }
    df_exercise['ExerciseType'] = df_exercise['ExerciseType'].apply(lambda x: id_to_exercise_type[x] if x in id_to_exercise_type else x)

    id_to_activity_type = {
        # 0: None, 
        1: 'BeforeBreakFast',
        2: 'AfterBreakFast',
        3: 'BeforeLunch',
        4: 'AfterLunch',
        5: 'BeforeDinner',
        6: 'AfterDinner',
        7: 'Bedtime',
        8: 'BeforeExercise',
        9: 'AfterExercise',
        12: 'Snack',
        14: 'Fasting',
        31: 'JustChecking',
    }
    df_exercise['ActivityType'] = df_exercise['ActivityTypeID'].apply(lambda x: id_to_activity_type[x] if x in id_to_activity_type else x)
    # df_exercise['ActivityType'].value_counts()
    print(df_exercise.shape, '<-- df_exercise.shape')

    # file = os.path.join(OneCohort_Args['FolderPath'], 'processed_RawFile_Exercise.csv')
    # df_exercise.to_csv(file, index=False)
    # RawName_to_dfRaw['Exercise'] = file
    RawName_to_dfRaw['Exercise'] = df_exercise

    return RawName_to_dfRaw


MetaDict = {
	"OneCohort_Args": OneCohort_Args,
	"SourceFile_SuffixList": SourceFile_SuffixList,
	"fill_missing_keys": fill_missing_keys,
	"replace_none_with_list": replace_none_with_list,
	"get_RawName_from_SourceFile": get_RawName_from_SourceFile,
	"get_InferenceEntry": get_InferenceEntry,
	"process_Source_to_Raw": process_Source_to_Raw
}