import os
import random
import numpy as np
import pandas as pd 
import random 
from datetime import datetime, timezone, timedelta



# --------------- Meta Info Functions ----------------
def MetaFn_for_BanditSMS_v250302(SPACE):
    
    Local_to_External_ModelSeries = {'SMSClickModel': 'SMSClickModel'}
    External_to_Local_ModelSeries = {v:k for k,v in Local_to_External_ModelSeries.items()}

    modelMetadata = []
    d = {}
    d['modelName'] = 'SMSClickModel'
    d['predictions'] = [ 'Default', 'Save', 'Pickup Reminder', 'Education'] 
    modelMetadata.append(d)

    metadata_response = {
        "body": {
            "modelMetadata":modelMetadata
        },
        "contentType": "application/json",
        "invokedProductionVariant": "AllTraffic"
    }
    meta_results = {
        # 'ModelSeries_to_ModelInstances': ModelSeries_to_ModelInstances,
        'Local_to_External_ModelSeries': Local_to_External_ModelSeries,
        'External_to_Local_ModelSeries': External_to_Local_ModelSeries,
        'modelMetadata': modelMetadata,
        'metadata_response': metadata_response,
    }
    return meta_results



# --------------- Meta Info Functions ----------------
def MetaFn_for_XgboostCTRSMS_v250302(SPACE):
    
    Local_to_External_ModelSeries = {'SMSModel-lightweight': 'SMSModel-lightweight'}
    External_to_Local_ModelSeries = {v:k for k,v in Local_to_External_ModelSeries.items()}

    modelMetadata = []
    d = {}
    d['modelName'] = 'SMSModel-lightweight'
    d['predictions'] = [ 'Default', 'Save', 'Pickup Reminder', 'Education'] 
    modelMetadata.append(d)

    metadata_response = {
        "body": {
            "modelMetadata":modelMetadata
        },
        "contentType": "application/json",
        "invokedProductionVariant": "AllTraffic"
    }
    meta_results = {
        # 'ModelSeries_to_ModelInstances': ModelSeries_to_ModelInstances,
        'Local_to_External_ModelSeries': Local_to_External_ModelSeries,
        'External_to_Local_ModelSeries': External_to_Local_ModelSeries,
        'modelMetadata': modelMetadata,
        'metadata_response': metadata_response,
    }
    return meta_results




# --------------- Meta Info Functions ----------------
def MetaFn_for_BanditSMS_v250302(SPACE):
    
    Local_to_External_ModelSeries = {'SMSClickModel': 'SMSClickModel'}
    External_to_Local_ModelSeries = {v:k for k,v in Local_to_External_ModelSeries.items()}

    modelMetadata = []
    d = {}
    d['modelName'] = 'SMSClickModel'
    d['predictions'] = [ 'Default', 'Save', 'Pickup Reminder', 'Education'] 
    modelMetadata.append(d)

    metadata_response = {
        "body": {
            "modelMetadata":modelMetadata
        },
        "contentType": "application/json",
        "invokedProductionVariant": "AllTraffic"
    }
    meta_results = {
        # 'ModelSeries_to_ModelInstances': ModelSeries_to_ModelInstances,
        'Local_to_External_ModelSeries': Local_to_External_ModelSeries,
        'External_to_Local_ModelSeries': External_to_Local_ModelSeries,
        'modelMetadata': modelMetadata,
        'metadata_response': metadata_response,
    }
    return meta_results



# --------------- Meta Info Functions ----------------
def MetaFn_for_RxEgm_v1122(SPACE):
    MODEL_ROOT = SPACE['MODEL_ROOT']
    MODEL_VERSION = SPACE['MODEL_VERSION']
    MODEL_VERSION_FOLDER = os.path.join(MODEL_ROOT, MODEL_VERSION)
    ModelSeries_to_ModelInstances = {}
    for model_series_name in os.listdir(os.path.join(MODEL_VERSION_FOLDER, 'models')):
        # os.listdir()
        model_series_folder = os.path.join(MODEL_VERSION_FOLDER, 'models', model_series_name)
        if not os.path.isdir(model_series_folder): continue
        ModelSeries_to_ModelInstances[model_series_name] = []
        
        for model_instance_name in os.listdir(model_series_folder):
            # model_instance_name = os.path.basename(model_series_folder)
            model_instance_folder = os.path.join(model_series_folder, model_instance_name)
            # print(model_instance_folder)
            if not os.path.isdir(model_instance_folder): continue
            ModelSeries_to_ModelInstances[model_series_name].append(model_instance_name)
    # pprint(ModelSeries_to_ModelInstances)
    Local_to_External_ModelSeries = {}

    for model_series_name in ModelSeries_to_ModelInstances:
        model_series_name_external = model_series_name.lower()
        Local_to_External_ModelSeries[model_series_name] = model_series_name_external
    External_to_Local_ModelSeries = {v:k for k,v in Local_to_External_ModelSeries.items()}
    
    shortName_to_fullName = {
        'Save': "Save",
        'Rmd': "Pickup Reminder",
        'Learn': "Education",
    }

    modelMetadata = []
    for ModelSeriesName, ModelInstanceList in ModelSeries_to_ModelInstances.items():
        modelseries_external = Local_to_External_ModelSeries[ModelSeriesName]
        
        d = {}
        d['modelName'] = modelseries_external
        d['predictions'] = []
        for ModelInstanceName in ModelInstanceList:
            modelinstance_external = ModelInstanceName.split('-')[2].split('.')[-1]
            fullName = shortName_to_fullName[modelinstance_external]
            d['predictions'].append(fullName)
            
        modelMetadata.append(d)
        
        
    metadata_response = {
        "body": {
            "modelMetadata":modelMetadata
        },
        "contentType": "application/json",
        "invokedProductionVariant": "AllTraffic"
    }
    meta_results = {
        'ModelSeries_to_ModelInstances': ModelSeries_to_ModelInstances,
        'Local_to_External_ModelSeries': Local_to_External_ModelSeries,
        'External_to_Local_ModelSeries': External_to_Local_ModelSeries,
        'modelMetadata': modelMetadata,
        'metadata_response': metadata_response,
    }
    return meta_results



def MetaFn_None(SPACE):
    return {}



# --------------- Meta Info Functions ----------------
def MetaFn_for_BanditSMS_v250225(SPACE):
    
    Local_to_External_ModelSeries = {'SMSModel': 'SMSModel'}
    External_to_Local_ModelSeries = {v:k for k,v in Local_to_External_ModelSeries.items()}

    modelMetadata = []
    d = {}
    d['modelName'] = 'SMSModel'
    d['predictions'] = [ 'Default', 'Save', 'Pickup Reminder', 'Education'] 
    modelMetadata.append(d)

    metadata_response = {
        "body": {
            "modelMetadata":modelMetadata
        },
        "contentType": "application/json",
        "invokedProductionVariant": "AllTraffic"
    }
    meta_results = {
        # 'ModelSeries_to_ModelInstances': ModelSeries_to_ModelInstances,
        'Local_to_External_ModelSeries': Local_to_External_ModelSeries,
        'External_to_Local_ModelSeries': External_to_Local_ModelSeries,
        'modelMetadata': modelMetadata,
        'metadata_response': metadata_response,
    }
    return meta_results


# --------------- TriggerName to CaseTriggerList ----------------
############################################
def TriggerFn_RxModel_v1122(inference_form):
    ########################### get the models to focus. 
    # for now, you will only focus on the first modelSeries. 
    ModelSeries_external_to_call = inference_form['models']
    ModelSeriesName = ModelSeries_external_to_call[0]
    ###########################
    
    ModelSeries_to_DrugNameList = {
        'jardiance': ['Jardiance'],
        'stiolto': ['Stiolto respimat'],
        'quviviq': ['Quviviq'],
    }
    TriggerName = 'AnyRx'
    drugName_list = ModelSeries_to_DrugNameList[ModelSeriesName]
    
    # drugName_list, TriggerName
    invitationId = inference_form['invitation']['invitationId']
    patientId    = inference_form['invitation']['patientId']
    createdDate = inference_form['invitation']['createdDate']
    
    rx_list = [rx for rx in inference_form['prescriptions'] 
               if rx['medication']['drugName'] in drugName_list]
    rx_case_list = [
        {
            'patient_id_encoded': patientId,
            'invitation_id_encoded': invitationId,
            'prescription_id_encoded': rx['prescriptionId'],
            'DT': createdDate, 
        } 
        for rx in rx_list
    ]
    TriggerName_to_CaseTriggerList = {
        TriggerName: rx_case_list
    }
    return  TriggerName_to_CaseTriggerList
############################################


def TriggerFn_AnyInv_v25025(inference_form):

    TriggerName = 'AnyInv'

    # drugName_list, TriggerName
    invitationId = inference_form['invitation']['invitationId']
    patientId    = inference_form['invitation']['patientId']
    createdDate  = inference_form['invitation']['createdDate']


    inv_info = {
        'patient_id_encoded': patientId,
        'invitation_id_encoded': invitationId,
        'DT': createdDate, 
    }

    TriggerName_to_CaseTriggerList = {
        TriggerName: [inv_info]
    }
    return  TriggerName_to_CaseTriggerList



# --------------- Post Process Functions ----------------
def PostFn_ModelInferenceInfo_to_final_result_EngagementPredToLabel(ModelCheckpointName_to_InferenceInfo):
    output = {"models": [], "status": {"code": 200, "message": "Success"}}

    for checkpoint, data in ModelCheckpointName_to_InferenceInfo.items():
        parts = checkpoint.split('-')

        # Extract model de`tails
        model_name = '-'.join(parts[:2])  # e.g., Jardiance/UniLabelPred-Jardiance.RxAf1w
        # date = parts[4]  # e.g., 2024.11.03
        # date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        date = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        version = parts[3]  # e.g., XGBClassifierV0.6
        prediction_name = parts[2].replace("Rx.", "")  # e.g., Rx.Learn
        score = data['y_pred_score_percentile'][0]

        # Find or create model entry
        model_entry = next((model for model in output["models"] if model["name"] == model_name), None)
        if not model_entry:
            model_entry = {"name": model_name, "date": date, "version": version, "predictions": [], "label": ""}
            output["models"].append(model_entry)

        # Append prediction details
        model_entry["predictions"].append({"name": prediction_name, "score": score})

    # Set the label as the name of the prediction with the highest score for each model
    for model in output["models"]:
        if model["predictions"]:
            max_prediction = max(model["predictions"], key=lambda x: x["score"])
            model["label"] = max_prediction["name"]

    return output



def PostFn_WithActionDict_v1121(ModelCheckpointName_to_InferenceInfo, SPACE):
    output = {"models": [], "status": {"code": 200, "message": "Success"}}

    for checkpoint, data in ModelCheckpointName_to_InferenceInfo.items():
        parts = checkpoint.split('-')

        # Extract model de`tails
        model_name = '-'.join(parts[:2])  # e.g., Jardiance/UniLabelPred-Jardiance.RxAf1w
        model_name = model_name.split('/')[0].lower()
        # date = parts[4]  # e.g., 2024.11.03
        # date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        date = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        # version = parts[3]  # e.g., XGBClassifierV0.6
        
        version = SPACE['MODEL_VERSION']
        prediction_name = parts[2].replace("Rx.", "")  # e.g., Rx.Learn
        score = data['y_pred_score_percentile'][0]

        # Find or create model entry
        model_entry = next((model for model in output["models"] if model["name"] == model_name), None)
        if not model_entry:
            model_entry = {"name": model_name, "date": date, "version": version, "predictions": [], "action": ""}
            output["models"].append(model_entry)

        # Append prediction details
        model_entry["predictions"].append({"name": prediction_name, "score": score})

    # Set the label as the name of the prediction with the highest score for each model
    for model in output["models"]:
        if model["predictions"]:
            max_prediction = max(model["predictions"], key=lambda x: x["score"])
            model["action"] = {'name': max_prediction["name"], 'score': max_prediction["score"]}

    return output





def PostFn_OptimalMessage_Bandit_v250225(ModelArtifactName_to_Inference, SPACE):

    # kwargs = os.environ.get('PostFn_kwargs', {})

    kwargs = {
        'use_multiple_treatment': False,
        'bandit_model_artifact_name': 'SMS_Bandit_vSampleData_ActionMessage__BanditV1__SMSBandit__2025.02.24__1a0aff0c8dbbfc71',
    }

    bandit_model_artifact_name = kwargs['bandit_model_artifact_name']
    model_artifact_name = bandit_model_artifact_name
    # model_artifact_name = 'SMS_Bandit_vSampleData_ActionMessage__BanditV1__SMSBandit__2025.02.24__1a0aff0c8dbbfc71'

    clientname_to_modelconfig = {
        'model': 'SMSModel', 
        'predictions': {
            'Default': {
                'key': 'action_prob_Default', 
                'model_artifact_name': model_artifact_name,
            },
            'Save': {
                'key': 'action_prob_Save',
                'model_artifact_name': model_artifact_name,
            },
            'Pickup Reminder': {
                'key': 'action_prob_Pickup Reminder',
                'model_artifact_name': model_artifact_name,
            },
            'Education': {
                'key': 'action_prob_Education',
                'model_artifact_name': model_artifact_name,
            },
        },
    }

    
    use_multiple_treatment = kwargs['use_multiple_treatment']

    if use_multiple_treatment:
        version = SPACE['MODEL_ENDPOINT'] + '__ABTest'
        model_name = clientname_to_modelconfig['model'] + '__ABTest'

    else:
        version = SPACE['MODEL_ENDPOINT']
        model_name = clientname_to_modelconfig['model']


    date = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    model_entry = {"name": model_name, "date": date, "version": version, "predictions": []}

    ma_output = ModelArtifactName_to_Inference[model_artifact_name]
    predictions_config = clientname_to_modelconfig['predictions']


    predictions = []
    for prediction_name, prediction_config in predictions_config.items():
        d = {}
        key = prediction_config['key']
        action_prob = ma_output[key]
        d['name'] = prediction_name
        d['score'] = round(100 * action_prob, 2)
        predictions.append(d)

    model_entry['predictions'] = predictions
    
    
    
    if use_multiple_treatment:
        random_score = random.uniform(0, 1)
        model_entry['random_score'] = random_score

        if 0 <= random_score <= 0.2:
            model_entry['action'] = {'name': 'Default', 'score': 100 * random_score}
            model_entry['action_reason'] = 'Treatment Group - Default'
        elif 0.2 < random_score <= 0.4:
            model_entry['action'] = {'name': 'Save', 'score': 100 * random_score}
            model_entry['action_reason'] = 'Treatment Group - Save'
        elif 0.4 < random_score <= 0.6:
            model_entry['action'] = {'name': 'Pickup Reminder', 'score': 100 * random_score}
            model_entry['action_reason'] = 'Treatment Group - Pickup Reminder'
        elif 0.6 < random_score <= 0.8:
            model_entry['action'] = {'name': 'Education', 'score': 100 * random_score}
            model_entry['action_reason'] = 'Treatment Group - Education'
        else:
            model_entry['action'] = {'name': ma_output['ActionName'][0], 'score': 100 * random_score}
            model_entry['action_reason'] = 'Treatment Group - AI Bandit Action'

    else:
        model_entry['action'] = {'name': ma_output['ActionName'][0], 'score': round(100 * ma_output['Reward'][0], 2)}
        model_entry['action_reason'] = 'Bandit Action'


    # model_entry['action'] = {'name'}
    del model_entry['action_reason']
    output = {"models": [model_entry], "status": {"code": 200, "message": "Success"}}
    return output


def PostFn_OptimalMessage_Bandit_ABTest_v250225(ModelArtifactName_to_Inference, SPACE):

    # kwargs = os.environ.get('PostFn_kwargs', {})

    kwargs = {
        'use_multiple_treatment': True,
        'bandit_model_artifact_name': 'SMS_Bandit_vSampleData_ActionMessage__BanditV1__SMSBandit__2025.02.24__1a0aff0c8dbbfc71',
    }

    bandit_model_artifact_name = kwargs['bandit_model_artifact_name']
    model_artifact_name = bandit_model_artifact_name
    # model_artifact_name = 'SMS_Bandit_vSampleData_ActionMessage__BanditV1__SMSBandit__2025.02.24__1a0aff0c8dbbfc71'

    clientname_to_modelconfig = {
        'model': 'SMSModel', 
        'predictions': {
            'Default': {
                'key': 'action_prob_Default', 
                'model_artifact_name': model_artifact_name,
            },
            'Save': {
                'key': 'action_prob_Save',
                'model_artifact_name': model_artifact_name,
            },
            'Pickup Reminder': {
                'key': 'action_prob_Pickup Reminder',
                'model_artifact_name': model_artifact_name,
            },
            'Education': {
                'key': 'action_prob_Education',
                'model_artifact_name': model_artifact_name,
            },
        },
    }

    
    use_multiple_treatment = kwargs['use_multiple_treatment']

    if use_multiple_treatment:
        version = SPACE['MODEL_ENDPOINT'] + '__ABTest'
        model_name = clientname_to_modelconfig['model'] + '__ABTest'

    else:
        version = SPACE['MODEL_ENDPOINT']
        model_name = clientname_to_modelconfig['model']


    date = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    model_entry = {"name": model_name, "date": date, "version": version, "predictions": []}

    ma_output = ModelArtifactName_to_Inference[model_artifact_name]
    predictions_config = clientname_to_modelconfig['predictions']


    predictions = []
    for prediction_name, prediction_config in predictions_config.items():
        d = {}
        key = prediction_config['key']
        action_prob = ma_output[key]
        d['name'] = prediction_name
        d['score'] = action_prob
        predictions.append(d)

    model_entry['predictions'] = predictions
    
    
    
    if use_multiple_treatment:
        random_score = random.uniform(0, 1)
        model_entry['random_score'] = random_score

        if 0 <= random_score <= 0.2:
            model_entry['action'] = 'Default'
            model_entry['action_reason'] = 'Treatment Group - Default'
        elif 0.2 < random_score <= 0.4:
            model_entry['action'] = 'Save'
            model_entry['action_reason'] = 'Treatment Group - Save'
        elif 0.4 < random_score <= 0.6:
            model_entry['action'] = 'Pickup Reminder'
            model_entry['action_reason'] = 'Treatment Group - Pickup Reminder'
        elif 0.6 < random_score <= 0.8:
            model_entry['action'] = 'Education'
            model_entry['action_reason'] = 'Treatment Group - Education'
        else:
            model_entry['action'] = ma_output['ActionName'][0]
            model_entry['action_reason'] = 'Treatment Group - AI Bandit Action'

    else:
        model_entry['action'] = ma_output['ActionName'][0]
        model_entry['action_reason'] = 'Bandit Action'


    output = {"models": [model_entry], "status": {"code": 200, "message": "Success"}}

    return output



# xgboost
# bandit
# from datetime import timezone 
# import random 

def PostFn_OptimalMessage_ExperimentalBanditVsXGBoost_v250226(ModelArtifactName_to_Inference, SPACE):

    # kwargs = os.environ.get('PostFn_kwargs', {})

    kwargs = {
        'use_multiple_treatment': True,
        'bandit_model_artifact_name': 'SMS_Bandit_vSampleData_ActionMessage__BanditV1__SMSBandit__2025.02.24__1a0aff0c8dbbfc71',
    }

    bandit_model_artifact_name = kwargs['bandit_model_artifact_name']
    model_artifact_name = bandit_model_artifact_name
    # model_artifact_name = 'SMS_Bandit_vSampleData_ActionMessage__BanditV1__SMSBandit__2025.02.24__1a0aff0c8dbbfc71'

    bandit_clientname_to_modelconfig = {
        'model': 'SMSModel', 
        'predictions': {
            'Default': {
                'key': 'action_prob_Default', 
                'model_artifact_name': model_artifact_name,
            },
            'Save': {
                'key': 'action_prob_Save',
                'model_artifact_name': model_artifact_name,
            },
            'Pickup Reminder': {
                'key': 'action_prob_Pickup Reminder',
                'model_artifact_name': model_artifact_name,
            },
            'Education': {
                'key': 'action_prob_Education',
                'model_artifact_name': model_artifact_name,
            },
        },
    }

    # {'CTR_SMS_Pred_All_Af0925__CTRinDefault_XGBClassifierV1__2025.02.25__98420b85d4c7581d': {'y_pred_score': array([0.48244783], dtype=float32),
    #                                                                                      'y_pred_score_percentile': array([19])},
    # 'CTR_SMS_Pred_All_Af0925__CTRinEdu_XGBClassifierV1__2025.02.25__89a96ecfb19156c1': {'y_pred_score': array([0.47922438], dtype=float32),
    #                                                                                     'y_pred_score_percentile': array([16])},
    # 'CTR_SMS_Pred_All_Af0925__CTRinRmd_XGBClassifierV1__2025.02.25__9bb36b94faf3877f': {'y_pred_score': array([0.43707955], dtype=float32),
    #                                                                                     'y_pred_score_percentile': array([39])},
    # 'CTR_SMS_Pred_All_Af0925__CTRinSave_XGBClassifierV1__2025.02.25__1d72ca92fdc1f339': {'y_pred_score': array([0.3878466], dtype=float32),
    #                                                                                     'y_pred_score_percentile': array([29])},


    default_xgboost_model_name = 'CTR_SMS_Pred_All_Af0925__CTRinDefault_XGBClassifierV1__2025.02.25__98420b85d4c7581d'
    edu_xgboost_model_name     = 'CTR_SMS_Pred_All_Af0925__CTRinEdu_XGBClassifierV1__2025.02.25__89a96ecfb19156c1'
    rmd_xgboost_model_name     = 'CTR_SMS_Pred_All_Af0925__CTRinRmd_XGBClassifierV1__2025.02.25__9bb36b94faf3877f'
    save_xgboost_model_name    = 'CTR_SMS_Pred_All_Af0925__CTRinSave_XGBClassifierV1__2025.02.25__1d72ca92fdc1f339'


    # xgboost_clientname_to_modelconfig = {
    #     'model': 'SMSModel', 
    #     'predictions': {
    #         'Default': {
    #             'key': 0, 
    #             'model_artifact_name': default_xgboost_model_name,
    #         },
    #         'Save': {
    #             'key': 0,
    #             'model_artifact_name': save_xgboost_model_name,
    #         },
    #         'Pickup Reminder': {
    #             'key': 0,
    #             'model_artifact_name': rmd_xgboost_model_name,
    #         },
    #         'Education': {
    #             'key': 0,
    #             'model_artifact_name': edu_xgboost_model_name,
    #         },
    #     },
    # }

    xgboost_model_name_to_predictionName = {
        default_xgboost_model_name: 'Default',
        save_xgboost_model_name: 'Save',
        rmd_xgboost_model_name: 'Pickup Reminder',
        edu_xgboost_model_name: 'Education',
    }

    
    use_multiple_treatment = kwargs['use_multiple_treatment']

    if use_multiple_treatment:
        version = SPACE['MODEL_ENDPOINT'] + '__ABTest'
        model_name = bandit_clientname_to_modelconfig['model'] + '__ABTest'

    else:
        version = SPACE['MODEL_ENDPOINT']
        model_name = bandit_clientname_to_modelconfig['model']


    date = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    model_entry = {"name": model_name, "date": date, "version": version, "predictions": []}



    if use_multiple_treatment:
        random_score = random.uniform(0, 1)
        model_entry['random_score'] = random_score
        if 0 < random_score < 1/3:
            random_score_v2 = random.uniform(0, 1)
            if 0 <= random_score_v2 <= 0.25:
                model_entry['action'] = {'name': 'Default', 'score': 100 * random_score}
                model_entry['action_reason'] = 'Treatment Group - Random - Default'
            elif 0.25 < random_score_v2 <= 0.5:
                model_entry['action'] = {'name': 'Save', 'score': 100 * random_score}
                model_entry['action_reason'] = 'Treatment Group - Random - Save'
            elif 0.5 < random_score_v2 <= 0.75:
                model_entry['action'] = {'name': 'Pickup Reminder', 'score': 100 * random_score}
                model_entry['action_reason'] = 'Treatment Group - Random - Pickup Reminder'
            elif 0.75 < random_score_v2 <= 1:
                model_entry['action'] = {'name': 'Education', 'score': 100 * random_score}
                model_entry['action_reason'] = 'Treatment Group - Random - Education'


            predictions_config = bandit_clientname_to_modelconfig['predictions']
            predictions = []
            for prediction_name, prediction_config in predictions_config.items():
                d = {}
                key = prediction_config['key']
                # action_prob = ma_output[key]
                d['name'] = prediction_name
                d['score'] = round(100 * 0, 2)
                predictions.append(d)
            model_entry['predictions'] = predictions
        
        
        
        elif 1/3 <= random_score <= 2/3:
            # bandit 
            ma_output = ModelArtifactName_to_Inference[bandit_model_artifact_name]
            model_entry['action'] = {'name': ma_output['ActionName'][0], 'score': 100 * random_score}
            model_entry['action_reason'] = 'Treatment Group - AI Bandit Action'

            # prepare prediction draft.
            predictions_config = bandit_clientname_to_modelconfig['predictions']
            predictions = []
            for prediction_name, prediction_config in predictions_config.items():
                d = {}
                key = prediction_config['key']
                action_prob = ma_output[key]
                d['name'] = prediction_name
                d['score'] = round(100 * action_prob, 2)
                predictions.append(d)
            model_entry['predictions'] = predictions
            

        elif 2/3 <= random_score <= 1:
            # xgboost

            # Filter only the XGBoost models (ones with y_pred_score)
            xgboost_models = {k: v for k, v in ModelArtifactName_to_Inference.items() if 'y_pred_score' in v}
            # Find the model with highest y_pred_score
            max_model = max(xgboost_models.items(), key=lambda x: x[1]['y_pred_score'][0])

            max_model_name = max_model[0]
            max_score = max_model[1]['y_pred_score'][0]
            name = xgboost_model_name_to_predictionName[max_model_name]
            model_entry['action'] = {'name': name, 'score': 100 * max_score}
            model_entry['action_reason'] = 'Treatment Group - XGBoost Action'

            predictions = []
            for xgboost_model_name, xgboost_model_output in xgboost_models.items():
                d = {}
                y_pred_score = xgboost_model_output['y_pred_score'][0]
                d['name'] = xgboost_model_name_to_predictionName[xgboost_model_name]
                d['score'] = round(100 * y_pred_score, 2)
                predictions.append(d)
            model_entry['predictions'] = predictions


    # model_entry['action'] = {'name'}
    # del model_entry['action_reason']
    output = {"models": [model_entry], "status": {"code": 200, "message": "Success"}}
    return output





def PostFn_OptimalMessage_Bandit_v250302(ModelArtifactName_to_Inference, SPACE):
    # print()
    model_artifact_name = [i for i in ModelArtifactName_to_Inference][0]
    assert 'Bandit' in model_artifact_name, f'model_artifact_name must contain "Bandit", but got {model_artifact_name}'
    bandit_model_artifact_name = model_artifact_name
    # model_artifact_name = 'SMS_Bandit_vSampleData_ActionMessage__BanditV1__SMSBandit__2025.02.24__1a0aff0c8dbbfc71'

    bandit_clientname_to_modelconfig = {
        'model': 'SMSClickModel', 
        'predictions': {
            'Default': {
                'key': 'action_prob_Default', 
                'model_artifact_name': model_artifact_name,
            },
            'Save': {
                'key': 'action_prob_Save',
                'model_artifact_name': model_artifact_name,
            },
            'Pickup Reminder': {
                'key': 'action_prob_Pickup Reminder',
                'model_artifact_name': model_artifact_name,
            },
            'Education': {
                'key': 'action_prob_Education',
                'model_artifact_name': model_artifact_name,
            },
        },
    }
    
    # use_multiple_treatment = kwargs['use_multiple_treatment']
    
    version = SPACE['MODEL_ENDPOINT']
    model_name = bandit_clientname_to_modelconfig['model']


    date = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    model_entry = {"name": model_name, "date": date, "version": version, "predictions": []}

    # bandit 
    ma_output = ModelArtifactName_to_Inference[bandit_model_artifact_name]
    model_entry['action'] = {'name': ma_output['ActionName'][0]}
    # model_entry['action_reason'] = 'Treatment Group - AI Bandit Action'

    # prepare prediction draft.
    predictions_config = bandit_clientname_to_modelconfig['predictions']
    predictions = []
    for prediction_name, prediction_config in predictions_config.items():
        d = {}
        key = prediction_config['key']
        action_prob = ma_output[key]
        assert len(action_prob) == 1, f'action_prob must be a single value, but got {action_prob}'
        action_prob = action_prob[0]
        d['name'] = prediction_name
        d['score'] = round(100 * action_prob, 2)

        if prediction_name == model_entry['action']['name']:
            model_entry['action']['score'] = round(100 * action_prob, 2)
        predictions.append(d)
    model_entry['predictions'] = predictions


    # model_entry['action'] = {'name'}
    # del model_entry['action_reason']
    output = {"models": [model_entry], "status": {"code": 200, "message": "Success"}}
    return output







def PostFn_OptimalMessage_XgboostCTR_v250302(ModelArtifactName_to_Inference, SPACE):

    # kwargs = os.environ.get('PostFn_kwargs', {})

    # kwargs = {
    #     'use_multiple_treatment': False,
    #     # 'bandit_model_artifact_name': 'SMS_Bandit_vSampleData_ActionMessage__BanditV1__SMSBandit__2025.02.24__1a0aff0c8dbbfc71',
    #     # 'bandit_model_artifact_name': 'SMS_Bandit_vSampleData_ActionMessage_NervousSystem__BanditV1__SMSBandit__2025.02.25__8e82e3175e021caf',
    # }

    default_xgboost_model_name = 'CTR_SMS_Pred_All_Af0925__CTRinDefault_XGBClassifierV1__2025.02.25__98420b85d4c7581d'
    edu_xgboost_model_name     = 'CTR_SMS_Pred_All_Af0925__CTRinEdu_XGBClassifierV1__2025.02.25__89a96ecfb19156c1'
    rmd_xgboost_model_name     = 'CTR_SMS_Pred_All_Af0925__CTRinRmd_XGBClassifierV1__2025.02.25__9bb36b94faf3877f'
    save_xgboost_model_name    = 'CTR_SMS_Pred_All_Af0925__CTRinSave_XGBClassifierV1__2025.02.25__1d72ca92fdc1f339'

    xgboost_model_name_to_predictionName = {
        default_xgboost_model_name: 'Default',
        save_xgboost_model_name: 'Save',
        rmd_xgboost_model_name: 'Pickup Reminder',
        edu_xgboost_model_name: 'Education',
    }

    
    version = SPACE['MODEL_ENDPOINT']
    model_name = 'SMSModel-lightweight' #  bandit_clientname_to_modelconfig['model']

    date = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    model_entry = {"name": model_name, "date": date, "version": version, "predictions": []}

    # Filter only the XGBoost models (ones with y_pred_score)
    xgboost_models = {k: v for k, v in ModelArtifactName_to_Inference.items() if 'y_pred_score' in v}
    # Find the model with highest y_pred_score
    max_model = max(xgboost_models.items(), key=lambda x: x[1]['y_pred_score'][0])

    max_model_name = max_model[0]
    max_score = max_model[1]['y_pred_score'][0]
    name = xgboost_model_name_to_predictionName[max_model_name]
    model_entry['action'] = {'name': name, 'score': round(100 * max_score, 2)}
    # model_entry['action_reason'] = 'Treatment Group - XGBoost Action'

    predictions = []
    for xgboost_model_name, xgboost_model_output in xgboost_models.items():
        d = {}
        y_pred_score = xgboost_model_output['y_pred_score'][0]
        d['name'] = xgboost_model_name_to_predictionName[xgboost_model_name]
        d['score'] = round(100 * y_pred_score, 2)
        predictions.append(d)
    model_entry['predictions'] = predictions


    # model_entry['action'] = {'name'}
    # del model_entry['action_reason']
    output = {"models": [model_entry], "status": {"code": 200, "message": "Success"}}
    return output





# ---------------- Choose Action Randomly with Treatment Design ----------------
def remove_single_value_brackets_from_nested_dict(nested_dict):
    for key, inner_dict in nested_dict.items():
        for inner_key, value in inner_dict.items():
            if isinstance(value, list) and len(value) == 1:
                inner_dict[inner_key] = value[0]
    return nested_dict


# Generate random probabilities ensuring they sum to 1
def generate_random_probabilities(action_list):
    probabilities = [random.random() for _ in action_list]
    total = sum(probabilities)
    normalized_probabilities = [round(p / total, 4) for p in probabilities]

    difference = 1 - sum(normalized_probabilities)
    normalized_probabilities[-1] += difference
    return normalized_probabilities

# Choose action based on probability
def choose_action_based_on_probability(probabilities):
    return random.choices(range(len(probabilities)), weights=probabilities)[0]

# Generate full structure
def generate_full_structure_with_action_reward(action_list):
    probabilities = generate_random_probabilities(action_list)
    action = action_list[choose_action_based_on_probability(probabilities)]
    reward = round(random.uniform(1, 10), 4)
    return {
        "Action": action,
        "Reward": reward,
        **{f"Action Probability {i}": probabilities[idx] for idx, i in enumerate(action_list)}
    }


def convert_modeloutput_to_optimaltiminglabels(x1, x2):
    global inference_results

    ModelCheckpointName_to_InferenceInfo = remove_single_value_brackets_from_nested_dict(x2)
    action_list = x1

    bandit_model_instance_list = [model_instance_name for model_instance_name in ModelCheckpointName_to_InferenceInfo if
                                  'Bandit' in model_instance_name]
    treatment_policy_list = ['random_policy'] + bandit_model_instance_list
    treatment_method = np.random.choice(treatment_policy_list, 1)[0]

    if treatment_method in ModelCheckpointName_to_InferenceInfo:
        inference_results = ModelCheckpointName_to_InferenceInfo[treatment_method]
    elif treatment_method == 'random_policy':
        inference_results = generate_full_structure_with_action_reward(action_list)

    print(ModelCheckpointName_to_InferenceInfo)

    action = int(inference_results['Action'])
    action_key = f"Action Probability {action}"
    probability = inference_results[action_key]
    current_time = datetime.now()
    current_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
    action_time = (current_time + timedelta(days=1)).replace(hour=int(action), minute=0, second=0,
                                                                  microsecond=0).strftime("%Y-%m-%d %H:%M:%S")

    return {
        "current_time": current_time_str,
        "model_name": treatment_method,
        "action_time": action_time,
        "action_key": action_key,
        "probability": probability
    }



#print(convert_modeloutput_to_optimaltiminglabels(sample_input.action_list,sample_input.ModelCheckpointName_to_InferenceInfo))
def pick_up_best_RxEngagement(ModelCheckpointName_to_InferenceInfo):
    
    # [i for i in ModelCheckpointName_to_InferenceInfo if '-Rx.' in i]
    UIModels = [i for i in ModelCheckpointName_to_InferenceInfo if '-Rx.' in i]

    action_to_UIModels = {
        [i for i in UIModel.split('-') if 'Rx.' in i][0].replace('Rx.', ''): UIModel for UIModel in UIModels
    }
    # action_to_UIModels
    
    action_to_score = {action: ModelCheckpointName_to_InferenceInfo[UIModel]['y_pred_score'][0] for action, UIModel in action_to_UIModels.items()}
    action_to_score

    max_key = max(action_to_score, key=action_to_score.get)

    print(max_key)
    action_output = {
        "action_UI": max_key,
        "model_name": action_to_UIModels[max_key],
        "score": action_to_score[max_key]
    }
    return action_output


def PostFn_NaiveForUniLabelPred(ModelArtifactName_to_Inference, SPACE):

    ModelArtifactName_to_Inference = {
        k: {k1: [round(float(i), 4) for i in list(v1)] for k1, v1 in v.items()} for k, v in ModelArtifactName_to_Inference.items()
    }

    return ModelArtifactName_to_Inference





NAME_TO_FUNCTION = {
    ####################### meta info
    'MetaFn_for_RxEgm_v1122': MetaFn_for_RxEgm_v1122,
    'MetaFn_for_BanditSMS_v250225': MetaFn_for_BanditSMS_v250225,
    'MetaFn_None': MetaFn_None,
    'MetaFn_for_XgboostCTRSMS_v250302': MetaFn_for_XgboostCTRSMS_v250302,
    'MetaFn_for_BanditSMS_v250302': MetaFn_for_BanditSMS_v250302,
    
    ####################### TriggerFn
    'TriggerFn_RxModel_v1122': TriggerFn_RxModel_v1122,
    'TriggerFn_AnyInv_v25025': TriggerFn_AnyInv_v25025,
    
    ####################### post process
    'PostFn_EngagementPredToLabel': PostFn_ModelInferenceInfo_to_final_result_EngagementPredToLabel,
    'PostFn_WithActionDict_v1121':  PostFn_WithActionDict_v1121,
    'PostFn_NaiveForUniLabelPred':  PostFn_NaiveForUniLabelPred,
    'PostFn_OptimalMessage_Bandit_v250225': PostFn_OptimalMessage_Bandit_v250225, 
    'PostFn_OptimalMessage_Bandit_ABTest_v250225': PostFn_OptimalMessage_Bandit_ABTest_v250225,
    'PostFn_OptimalMessage_ExperimentalBanditVsXGBoost_v250226': PostFn_OptimalMessage_ExperimentalBanditVsXGBoost_v250226,
    'PostFn_OptimalMessage_XgboostCTR_v250302': PostFn_OptimalMessage_XgboostCTR_v250302,
    'PostFn_OptimalMessage_Bandit_v250302': PostFn_OptimalMessage_Bandit_v250302,

}

