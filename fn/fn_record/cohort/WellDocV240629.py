import os

import pandas as pd

import numpy as np

OneCohort_Args = {'CohortLabel': 4,
 'CohortName': 'WellDoc2025ALS',
 'FolderPath': './_Data/0-Data_Raw/WellDoc2025ALS/',
 'SourcePath': './_Data/0-Data_Raw/WellDoc2025ALS/',
 'Source2CohortName': 'WellDocV240629'}

SourceFile_SuffixList = ['csv']

def get_RawName_from_SourceFile(file_path, OneCohort_Args):
    """
    Extracts a 'raw name' from a given file path.

    This function takes a file path and extracts what is assumed to be a 'raw name'
    by splitting the path and selecting specific parts. The 'raw name' is considered
    to be the last part of the file name before the file extension.

    Args:
        file_path (str): The full path of the file from which to extract the raw name.
        OneCohort_Args: Currently unused. Reserved for future functionality.

    Returns:
        str: The extracted 'raw name' from the file path.

    """
    RawName = file_path.split('_')[-1].split('.')[0]
    return RawName


def process_Source_to_Raw(OneCohort_Args, SourceFile_List, get_RawName_from_SourceFile,SPACE):
    """
        Process source files to raw data files, including renaming columns and merging certain files.

        Args:
        OneCohort_Args (dict): Dictionary containing processing arguments, including 'FolderPath'.
        SourceFile_List (list): List of source file paths.
        get_RawName_from_SourceFile (function): Function to extract raw name from file path.

        Returns:
        dict: Mapping of raw names to processed file paths.
        """
    # Initialize dictionary to store raw names and their corresponding file paths
    RawName_to_dfRaw = {}
    for file_path in SourceFile_List:
        # Extract the raw name for each file using the function 
        RawName = get_RawName_from_SourceFile(file_path, OneCohort_Args)
        # Assign value file_path to key RawName
        RawName_to_dfRaw[RawName] = file_path


    # ---------- update the PatientID
    # Process files to update "PatientId" to "PatientID"
    # loop RawName_to_dfRaw dictionary 
    for RawName, file_path in RawName_to_dfRaw.items():
        # Skip empty files
        if os.path.getsize(file_path) == 0: continue  
        # Read only the header of the file to check columns
        df = pd.read_csv(file_path, nrows = 0)
        # Skip files without "PatientId" column
        if 'PatientId' not in df.columns: continue 
        # Define the new file path for the processed file
        file_path_new = os.path.join(OneCohort_Args['FolderPath'], f'processed_RawFile_{RawName}.csv')
        RawName_to_dfRaw[RawName] = file_path_new
        if not os.path.exists(file_path_new): 
            # RawName_to_dfRaw.pop(RawName, None)
            df = pd.read_csv(file_path)
            df = df.rename(columns={'PatientId': 'PatientID'})
            df.to_csv(file_path, index=False)
            # print(f'processed file: {file_path_new}')

    # ---------- merge UserDetail and Patient for Patient
    file = os.path.join(OneCohort_Args['FolderPath'], 'processed_RawFile_Ptt.csv')
    RawName_to_dfRaw.pop('UserDetail', None)
    RawName_to_dfRaw.pop('Patient', None)
    UserDetail_file = [i for i in SourceFile_List 
                        if get_RawName_from_SourceFile(i, OneCohort_Args) == 'UserDetail' and 'processed' not in i][0]
    df_UserDetail = pd.read_csv(UserDetail_file)    
    print(UserDetail_file)
    df_UserDetail = df_UserDetail.rename(columns={'UserID': 'PatientID'})
    print(df_UserDetail.columns)

    Patient_file = [i for i in SourceFile_List 
                    if get_RawName_from_SourceFile(i, OneCohort_Args) == 'Patient' and 'processed' not in i][0]
    print(Patient_file)
    df_Patient = pd.read_csv(Patient_file)
    print(df_Patient.columns)

    df_Ptt = pd.merge(df_Patient, df_UserDetail, on='PatientID', how='outer')

    df_Ptt.to_csv(file, index=False)
    RawName_to_dfRaw['Ptt'] = file


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
    df_food_path = [i for i in SourceFile_List if 'ELogFoodItem' in i][0]
    df_food = pd.read_csv(df_food_path, low_memory=False)
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


    df_carbs_path = [i for i in SourceFile_List if 'ELogCarbsEntry' in i][0]
    df_carbs = pd.read_csv(df_carbs_path, low_memory=False)
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

    df_carbs = df_carbs[columns]
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

    file = os.path.join(OneCohort_Args['FolderPath'], 'processed_RawFile_Diet.csv')
    df_meal.to_csv(file, index=False)
    RawName_to_dfRaw['Diet'] = file



    ############################ add the medication ##############################
    # import pandas as pd
    # pd.set_option('display.max_columns', None)

    medadmin_path = [i for i in SourceFile_List if 'MedAdmin' in i][0]
    print(medadmin_path)
    df_med = pd.read_csv(medadmin_path, low_memory=False)
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
    df_med = df_med[columns].reset_index()
    file = os.path.join(OneCohort_Args['FolderPath'], 'processed_RawFile_MedAdmin.csv')
    df_med.to_csv(file, index=False)
    RawName_to_dfRaw['MedAdmin'] = file


    ############################# add the exercise #############################
    exercise_path = [i for i in SourceFile_List if 'ELogExercise' in i][0]
    print(exercise_path)
    df_exercise = pd.read_csv(exercise_path, low_memory=False)
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

    df_exercise = df_exercise[columns]
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

    file = os.path.join(OneCohort_Args['FolderPath'], 'processed_RawFile_Exercise.csv')
    df_exercise.to_csv(file, index=False)
    RawName_to_dfRaw['Exercise'] = file

    return RawName_to_dfRaw


MetaDict = {
	"OneCohort_Args": OneCohort_Args,
	"SourceFile_SuffixList": SourceFile_SuffixList,
	"get_RawName_from_SourceFile": get_RawName_from_SourceFile,
	"process_Source_to_Raw": process_Source_to_Raw
}