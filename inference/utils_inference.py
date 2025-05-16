import numpy as np 
import pandas as pd
import logging 
import copy
import os
from datetime import datetime 
from datasets.fingerprint import Hasher 
from flask import Flask, request, jsonify, Response


logger = logging.getLogger(__name__)

Record_Proc_Config = {
    'save_data': False, 
    'load_data':False, 
    'via_method': 'df'
}

Case_Proc_Config = {
    'max_trigger_case_num': None, 
    'use_task_cache': True, 
    'caseset_chunk_size': 50000,
    'save_data': False, 
    'load_data': False, 
    'load_casecollection': False,
    'via_method': 'df',
    'n_cpus': 1, 
    'batch_size': None,  
}

OneEntryArgs_items_for_inference = ['Input_Part']




def load_Inference_Entry_Example(INF_CohortName, 
                                 CohortName_to_OneCohortArgs,
                                 Cohort,
                                 CohortFn,
                                 SPACE):
    ################
    OneCohort_Args = CohortName_to_OneCohortArgs[INF_CohortName].copy()

    Source2CohortName = OneCohort_Args['Source2CohortName']
    cohort_fn = CohortFn(Source2CohortName, SPACE)
    cohort = Cohort(OneCohort_Args, SPACE, cohort_fn)
    cohort.setup_fn(cohort_fn)
    cohort.initialize_cohort(save_data = False, load_data = False)

    # Get Inference_Entry
    SourceFile_List = cohort.SourceFile_List
    OneCohort_Args = cohort.OneCohort_Args
    get_RawName_from_SourceFile = cohort.get_RawName_from_SourceFile
    get_InferenceEntry = cohort.cohort_fn.get_InferenceEntry

    Inference_Entry = get_InferenceEntry(OneCohort_Args, 
                                         SourceFile_List, 
                                         get_RawName_from_SourceFile)

    return Inference_Entry
    

def process_CaseBaseArgs_ForInference(CaseBaseArgs, InputCFArgs_ForInference = None):
    CaseBaseArgs_ForInference = {}

    OriginalTriggerArgs = CaseBaseArgs['Trigger']
    
    OriginalTriggerArgs = OriginalTriggerArgs.copy()
    
    for i in ['TagRec', 'Filter', 'Group']:
        if i in OriginalTriggerArgs: del OriginalTriggerArgs[i]
            
    for CaseCollectionName, CaseCollectionArgs in CaseBaseArgs.items():
        ObsTask = CaseCollectionArgs.get('ObsTask', {})
    
    # now you have ObsTask from the last CaseCollectionArgs 
    ObsTask = ObsTask.copy()

    CF_list = []
    for i in ['TagCF_list']:
        ObsTask['TagCF_list'] = []
        if type(InputCFArgs_ForInference) == list:
            
            CF_list = InputCFArgs_ForInference
            ObsTask['CF_list'] = CF_list

        elif type(InputCFArgs_ForInference) == dict:
            CF_list = InputCFArgs_ForInference['CF_list']
            ObsTask['CF_list'] = CF_list

        else:
            logger.info(f'Currently only support list of CFs: current version {type(InputCFArgs_ForInference)}')
            raise ValueError(f'Currently only support list of CFs: current version {type(InputCFArgs_ForInference)}')

    OriginalTriggerArgs['ObsTask'] = ObsTask    
    CaseBaseArgs_ForInference['Trigger'] = OriginalTriggerArgs

    results = {
        'CaseBaseArgs_ForInference': CaseBaseArgs_ForInference,
        'CF_list': CF_list,
    }
    return results


def get_complete_InfoSettings(model_base, CohortName_list, InputCFArgs_ForInference = None):

    TriggerCaseBaseName_to_TriggerCaseBaseArgs = {}
    AIDataName_to_AIDataArgs = {}

    CF_list_ForInference_total = []
    for model_artifact_name, ModelInfo in model_base.ModelArtifactName_to_ModelInfo.items():


        model_artifact = ModelInfo['model_artifact']    
        aidata = model_artifact.aidata 


        OneAIDataArgs = aidata.OneAIDataArgs.copy()
        # TriggerCaseBaseNameRaw = aidata.OneAIDataArgs['TriggerCaseBaseName']
        TriggerCaseBaseArgsRaw = OneAIDataArgs['TriggerCaseBaseArgs']
        # TriggerCaseBaseNameRaw_to_TriggerCaseBaseArgsRaw[TriggerCaseBaseNameRaw] = TriggerCaseBaseArgsRaw

        ########################################
        if InputCFArgs_ForInference is None:
            InputCFArgs_ForInference = aidata.get_CFs_ForInference()

        # CF_list_ForInference_total += CFArgs_ForInference
        results = process_CaseBaseArgs_ForInference(TriggerCaseBaseArgsRaw, InputCFArgs_ForInference)
        TriggerCaseBaseArgs = results['CaseBaseArgs_ForInference']
        CF_list = results['CF_list']
        CF_list_ForInference_total += CF_list


        TriggerName = TriggerCaseBaseArgs['Trigger']['TriggerName']
        TriggerCaseBaseName = f'CaseBase-{TriggerName}-' + Hasher().hash(TriggerCaseBaseArgs)
        TriggerCaseBaseName_to_TriggerCaseBaseArgs[TriggerCaseBaseName] = TriggerCaseBaseArgs
        # TriggerCaseBaseNameRaw_to_TriggerCaseBaseName[TriggerCaseBaseNameRaw] = TriggerCaseBaseName
        ########################################

        OneAIDataArgs  = aidata.OneAIDataArgs.copy()# ['NeatArgs'].copy() 
        Input_Part = OneAIDataArgs['OneEntryArgs']['Input_Part']
        Input_Part['InputCFs_Args'] = InputCFArgs_ForInference
        OneAIDataArgs['OneEntryArgs'] = {'Input_Part': Input_Part}  

        OneAIDataName  = aidata.OneAIDataName
        Name_to_Data   = aidata.Name_to_Data
        OneAIDataArgs['CohortName_list'] = CohortName_list
        OneAIDataArgs['TriggerCaseBaseName'] = TriggerCaseBaseName   
        OneAIDataArgs['TriggerCaseBaseArgs'] = TriggerCaseBaseArgs 
        OneAIDataArgs['Name_to_Data'] = Name_to_Data
        AIDataName_to_AIDataArgs[OneAIDataName] = OneAIDataArgs
    
    CF_list_ForInference_total = list(set(CF_list_ForInference_total))  
    Ckpd_to_CkpdObsConfig_total = {}
    CF_to_CFArgs_total = {}
    TagCF_to_TagCFArgs_total = {}


    for model_artifact_name, ModelInfo in model_base.ModelArtifactName_to_ModelInfo.items():

        model_artifact = ModelInfo['model_artifact']    
        # model_instance = modelinstanceinfo['model_instance']    
        aidata = model_artifact.aidata 
        Case_Args_Settings   = aidata.OneAIDataArgs['Case_Args_Settings']
        Ckpd_to_CkpdObsConfig = Case_Args_Settings['Ckpd_to_CkpdObsConfig']
        CF_to_CFArgs = Case_Args_Settings['CF_to_CFArgs']
        TagCF_to_TagCFArgs = Case_Args_Settings['TagCF_to_TagCFArgs']
        
        for CkpdName, CkpdObsConfig in Ckpd_to_CkpdObsConfig.items():
            if CkpdName in Ckpd_to_CkpdObsConfig_total:
                assert Hasher().hash(CkpdObsConfig) == Hasher().hash(Ckpd_to_CkpdObsConfig_total[CkpdName])
            else:
                Ckpd_to_CkpdObsConfig_total[CkpdName] = CkpdObsConfig
        
        for CFName, CFArgs in CF_to_CFArgs.items():
            if CFName in CF_to_CFArgs_total:
                assert Hasher().hash(CFArgs) == Hasher().hash(CF_to_CFArgs_total[CFName])
            else:
                CF_to_CFArgs_total[CFName] = CFArgs
                
        
        for TagCFName, TagCFArgs in TagCF_to_TagCFArgs.items():
            if TagCFName in TagCF_to_TagCFArgs_total:
                assert Hasher().hash(TagCFArgs) == Hasher().hash(TagCF_to_TagCFArgs_total[TagCFName])
            else:
                TagCF_to_TagCFArgs_total[TagCFName] = TagCFArgs
            
    GROUP_TO_GROUPMethodArgs =  {
        'GrpBase': {
            'GroupName': 'GrpBase', 
            'GroupCategoryName_list': ['Base'],
            'ROName_to_RONameArgs': {},
        },
    }
         
    Case_Args_Settings = {
        'Ckpd_to_CkpdObsConfig': Ckpd_to_CkpdObsConfig_total,
        'GROUP_TO_GROUPMethodArgs': GROUP_TO_GROUPMethodArgs,
        'CF_to_CFArgs': CF_to_CFArgs_total,
        'TagCF_to_TagCFArgs': TagCF_to_TagCFArgs_total,
    }

    # pprint(Case_Args_Settings, sort_dicts=False)
    InfoSettings = {}
    InfoSettings['TriggerCaseBaseName_to_TriggerCaseBaseArgs'] = TriggerCaseBaseName_to_TriggerCaseBaseArgs
    InfoSettings['OneAIDataName_to_OneAIDataArgs'] = AIDataName_to_AIDataArgs
    InfoSettings['Case_Args_Settings'] = Case_Args_Settings
    InfoSettings['CF_list_ForInference'] = CF_list_ForInference_total
    return InfoSettings


def load_AIData_Model_InfoSettings( 
        ModelEndpoint_Path = None,
        InputCFArgs_ForInference = None, 
        InferenceArgs = None, 
        INF_CohortName = None,    
        INF_OneCohortArgs = None,         
        Record_Proc_Config = None,
        Case_Proc_Config = None,
        OneEntryArgs_items_for_inference = None,
        get_ROCOGammePhiInfo_from_CFList = None, 
        load_model_instance_from_nn = None, 
        AIData_Base = None,
        Model_Base = None, 
        SPACE = None):

    # ---------- model_base --------- (from one ModelVersion)
    if ModelEndpoint_Path is None: ModelEndpoint_Path = os.path.join(SPACE['MODEL_ROOT'], SPACE['MODEL_ENDPOINT'])
    model_base = Model_Base(
        ModelEndpoint_Path = ModelEndpoint_Path,
        load_model_instance_from_nn = load_model_instance_from_nn,
        SPACE = SPACE,
    )

    # ---------- aidata_base ---------
    InfoSettings = get_complete_InfoSettings(model_base, INF_CohortName, InputCFArgs_ForInference)
    OneAIDataName_to_OneAIDataArgs = InfoSettings['OneAIDataName_to_OneAIDataArgs']
    CohortName_list = [INF_CohortName]
    aidata_base = AIData_Base(
        OneAIDataName_to_OneAIDataArgs = OneAIDataName_to_OneAIDataArgs,
        OneEntryArgs_items_for_inference = OneEntryArgs_items_for_inference,
        CohortName_list_for_inference = CohortName_list, 
        SPACE = SPACE, 
    )

    # ---------- update InfoSettings -----------
    Case_Args_Settings = InfoSettings['Case_Args_Settings']
    CF_to_CFArgs = Case_Args_Settings['CF_to_CFArgs']
    CF_list_ForInference = InfoSettings['CF_list_ForInference']
    
    ROCOGammaPhiInfo = get_ROCOGammePhiInfo_from_CFList(CF_list_ForInference, CF_to_CFArgs)
    HumanRecordRecfeat_Args = ROCOGammaPhiInfo['HumanRecordRecfeat_Args']

    TriggerCaseBaseName_List = list(set([v['TriggerCaseBaseName'] for k, v in OneAIDataName_to_OneAIDataArgs.items()]))
    TriggerCaseBaseName_to_CohortNameList = {}
    for TriggerCaseBaseName in TriggerCaseBaseName_List:
        TriggerCaseBaseName_to_CohortNameList[TriggerCaseBaseName] = CohortName_list
        
    InfoSettings.update({
        'InferenceArgs': InferenceArgs,
        'CF_list': CF_list_ForInference,
        'InputCFArgs_ForInference': InputCFArgs_ForInference,

        'INF_CohortName': INF_CohortName,
        'INF_OneCohortArgs': INF_OneCohortArgs,
        'CohortName_to_OneCohortArgs': {INF_CohortName: INF_OneCohortArgs},

        'ROCOGammaPhiInfo': ROCOGammaPhiInfo,
        'HumanRecordRecfeat_Args': HumanRecordRecfeat_Args,
        'TriggerCaseBaseName_to_CohortNameList': TriggerCaseBaseName_to_CohortNameList,
        
        'CohortName_list': CohortName_list,
        'Record_Proc_Config': Record_Proc_Config,
        'Case_Proc_Config': Case_Proc_Config,
    })

    Context = {
        'model_base': model_base,
        'aidata_base': aidata_base,
        'InfoSettings': InfoSettings,
    }

    if 'MODEL_VERSION' in SPACE:
        SPACE['MODEL_ENDPOINT'] = SPACE['MODEL_VERSION']

    return Context





def pipeline_inference_for_modelbase(Inference_Entry,
                                     Record_Base,
                                     Case_Base,
                                     aidata_base, 
                                     model_base,
                                     InfoSettings, 
                                     SPACE):
    # --------- record_base ---------
    s = datetime.now()
    CohortName_list = InfoSettings['CohortName_list']
    HumanRecordRecfeat_Args = InfoSettings['HumanRecordRecfeat_Args']
    CohortName_to_OneCohortArgs = InfoSettings['CohortName_to_OneCohortArgs']
    TriggerCaseBaseName_to_TriggerCaseBaseArgs = InfoSettings['TriggerCaseBaseName_to_TriggerCaseBaseArgs']
    
    record_base = Record_Base(CohortName_list, 
                                HumanRecordRecfeat_Args,
                                CohortName_to_OneCohortArgs,
                                SPACE = SPACE, 
                                Inference_Entry = Inference_Entry,
                                Record_Proc_Config = Record_Proc_Config,
                                )
    e = datetime.now()
    du1 = e-s
    
    
    # --------- record_base ---------
    s = datetime.now()
    TriggerCaseBaseName_to_CohortNameList = InfoSettings['TriggerCaseBaseName_to_CohortNameList']
    Case_Proc_Config = InfoSettings['Case_Proc_Config']
    Case_Args_Settings = InfoSettings['Case_Args_Settings']
    case_base = Case_Base(
        record_base = record_base, 
        TriggerCaseBaseName_to_CohortNameList = TriggerCaseBaseName_to_CohortNameList, 
        TriggerCaseBaseName_to_TriggerCaseBaseArgs = TriggerCaseBaseName_to_TriggerCaseBaseArgs,
        Case_Proc_Config = Case_Proc_Config,
        Case_Args_Settings = Case_Args_Settings, 
    )
    e = datetime.now()
    du2 = e-s
    
    
    # --------- aidata_base ---------
    s = datetime.now()
    aidata_base.update_CaseBase(case_base)
    e = datetime.now()
    du3 = e-s
    
    
    # --------- model_base ---------
    s = datetime.now()
    InferenceArgs = InfoSettings['InferenceArgs']
    model_base.aidata_base = aidata_base
    ModelArtifactName_to_ModelInfo = model_base.ModelArtifactName_to_ModelInfo
    ModelArtifactName_to_Inference = {}
    for model_artifact_name, ModelInfo in ModelArtifactName_to_ModelInfo.items():
        model_artifact = ModelInfo['model_artifact']
        OneAIDataName = model_artifact.aidata.OneAIDataName
        aidata = aidata_base.get_aidata_from_OneAIDataName(OneAIDataName)
        Name = [i for i in aidata.Name_to_Data][0]
        Data = aidata.Name_to_Data[Name]
        inference = model_artifact.inference(Data, InferenceArgs)
        assert model_artifact_name == model_artifact.model_checkpoint_name
        ModelArtifactName_to_Inference[model_artifact_name] = inference
        
    e = datetime.now()
    du4 = e-s
    
    
    total_time = du1 + du2 + du3 + du4
    inference_results = {
        'ModelArtifactName_to_Inference': ModelArtifactName_to_Inference, 
        'record_base': record_base,
        'case_base': case_base,
        'aidata_base': aidata_base,
        'total_time': total_time,
        'du1': du1,
        'du2': du2,
        'du3': du3,
        'du4': du4,
    }
    return inference_results
    

def get_prediction_response(inference_form, **kwargs):
    ############# prepare your kwargs #############
    meta_results            = kwargs['meta_results']
    Inference_Entry_Example = kwargs['Inference_Entry_Example']

    Record_Base = kwargs['Record_Base']
    Case_Base   = kwargs['Case_Base']
    aidata_base = kwargs['aidata_base']
    model_base  = kwargs['model_base']

    InfoSettings = kwargs['InfoSettings']
    LoggerLevel = kwargs['LoggerLevel']
    # pipeline_inference_for_modelbase = kwargs['pipeline_inference_for_modelbase']

    TrigFn = kwargs['TrigFn']
    PostFn = kwargs['PostFn']
    SPACE  = kwargs['SPACE']
    fill_missing_keys = kwargs['fill_missing_keys']
    TrigFnArgs = kwargs.get('TrigFnArgs', {})
    PostFnArgs = kwargs.get('PostFnArgs', {})
    #############################################

    request_type = inference_form.get('requestType', None)
    if request_type == 'metadata':
        return meta_results['metadata_response']['body'], 200
        
    # get ModelSeries_external_to_call
    ModelSeries_external_to_call = inference_form['models']
    External_to_Local_ModelSeries = meta_results['External_to_Local_ModelSeries']
    ModelSeries_to_call = [External_to_Local_ModelSeries[i] for i in ModelSeries_external_to_call 
                            if i in External_to_Local_ModelSeries]

    if len(ModelSeries_to_call) == 0:
        results = {
            "status": {
                "code": 500,
                "message": f"No model to call in current ModelSeries: {External_to_Local_ModelSeries}",
            }
        }
        return results, 500


    # get TriggerName_to_CaseTriggerList
    #########################################################
    TriggerName_to_CaseTriggerList = TrigFn(inference_form, **TrigFnArgs)
    TriggerName_to_dfCaseTrigger = {k: pd.DataFrame(v) for k, v in TriggerName_to_CaseTriggerList.items()}
    #########################################################


    if sum([len(v) for v in TriggerName_to_CaseTriggerList.values()]) == 0:
        results = {
            "status": {
                "code": 500,
                "message": f"No prediction to trigger in current TriggerName_to_CaseTriggerList: {TriggerName_to_CaseTriggerList}",
            }
        }
        return results, 500

    # prepare Inference Entry
    template_form = Inference_Entry_Example['template_form']
    inference_form = fill_missing_keys(inference_form, template_form)
            
    Inference_Entry = {}
    Inference_Entry['inference_form'] = inference_form
    Inference_Entry['template_form']  = template_form
    Inference_Entry['TriggerName_to_dfCaseTrigger'] = TriggerName_to_dfCaseTrigger
    Inference_Entry['ModelSeries_to_call'] = ModelSeries_to_call
    
    # --------- pipeline_inference_for_modelbase ---------
    inference_results = pipeline_inference_for_modelbase(
        Inference_Entry = Inference_Entry,
        Record_Base = Record_Base, 
        Case_Base = Case_Base,
        aidata_base = aidata_base, 
        model_base = model_base,
        InfoSettings = InfoSettings, 
        SPACE = SPACE
    )
    
    ModelArtifactName_to_Inference = inference_results['ModelArtifactName_to_Inference']
    
    if LoggerLevel == 'INFO': 
        # ----------------------------------------------------
        du1 = inference_results['du1']
        du2 = inference_results['du2']
        du3 = inference_results['du3']
        du4 = inference_results['du4']
        total_time = inference_results['total_time']
        logger.info(ModelArtifactName_to_Inference)
        logger.info(f'record_base: {du1}')
        logger.info(f'case_base: {du2}')
        logger.info(f'aidata_base and model_base update: {du3}')
        logger.info(f'model_infernece: {du4}')
        logger.info(f'total_time: {total_time}')
        
    #########################################################
    results = PostFn(ModelArtifactName_to_Inference, SPACE, **PostFnArgs)
    #########################################################

    # make sure your output is jsonify.
    results_and_status = results, 200
    return results_and_status