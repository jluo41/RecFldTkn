import os
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from datetime import datetime 
from datasets.fingerprint import Hasher 
import json 
import logging
from nn.eval.binarypred import BinaryPredEval
from recfldtkn.aidata_base.aidata import AIData
import bisect
import xgboost as xgb
import numpy as np 


logger = logging.getLogger(__name__)


def in_training_visualization(model, ModelArgs, TrainingArgs, Name_to_Data):
    # evaluation during the training. 

    # aidata = self.aidata
    # Name_to_DsAIData = aidata.Name_to_DsAIData
    set_names = list(Name_to_Data.keys())   
    # ModelArgs = self.ModelArgs
    # TrainingArgs = self.TrainingArgs
    
    evals_result = model.evals_result()
    results = [v for k, v in evals_result.items()]
    evals_result = dict(zip(set_names, results))

    algorithm_name = ModelArgs['algorithm_name']
    eval_metric = TrainingArgs['eval_metric']
    epochs = len(evals_result[set_names[0]][eval_metric])
    x_axis = range(0, epochs)
    fig, ax = plt.subplots()
    for splitname in evals_result:
        ax.plot(x_axis, evals_result[splitname]['logloss'], label=splitname)
    ax.legend()
    plt.ylabel('Log Loss')
    plt.title(f'{algorithm_name} Log Loss')
    plt.show()
    
    
def get_percentile(y_pred_score, sorted_pred_score):
    index = bisect.bisect_left(sorted_pred_score, y_pred_score)
    percentile = int(round((index + 0.5) / len(sorted_pred_score) * 100, 0))
    return percentile



class XGBClassifierInstance:
    def __init__(self, 
                 ModelNames = None, 
                 aidata = None,
                 ModelArgs = None, 
                 TrainingArgs = None, 
                 InferenceArgs = None, 
                 EvaluationArgs = None,
                 SPACE = None, 
                 ):
        
        # self.ModelName = ModelName
        # self.ModelVersion = ModelVersion
        self.ModelNames = ModelNames
        self.aidata = aidata
        self.ModelArgs = ModelArgs
        self.TrainingArgs = TrainingArgs
        self.InferenceArgs = InferenceArgs
        self.EvaluationArgs = EvaluationArgs
        self.SPACE = SPACE

        self.ModelInstanceArgs = {
            'ModelArgs': ModelArgs,
            'TrainingArgs': TrainingArgs,
            'InferenceArgs': InferenceArgs,
            'EvaluationArgs': EvaluationArgs,
            'SPACE': SPACE,
        }

        self.model_checkpoint_name = self.ModelNames['model_checkpoint_name']
        self.MODEL_ENDPOINT = self.ModelNames['MODEL_ENDPOINT']
        self.model_checkpoint_path = os.path.join(SPACE['MODEL_ROOT'], 
                                                  self.MODEL_ENDPOINT, 
                                                  'models', 
                                                  self.model_checkpoint_name, 
                                                  )

    def init_model(self):
        ModelArgs = self.ModelArgs
        TrainingArgs = self.TrainingArgs
        # model = init_xgboost_models(ModelArgs, TrainingArgs)
        args = {k: v for k, v in ModelArgs.items() if k != 'algorithm_name'}
        model = xgb.XGBClassifier(**args, **TrainingArgs)
    
        self.model = model 

        self.wv = None
    
    def save_model(self, model_checkpoint_path = None):

        if model_checkpoint_path is None:
            model_checkpoint_path = self.model_checkpoint_path

        # -----------  save aidata -----------
        data_path = os.path.join(model_checkpoint_path, 'Data')
        if not os.path.exists(data_path): os.makedirs(data_path)
        aidata = self.aidata
        save_args = {'sample_num': 5}
        aidata.save_aidata(data_path, save_args)

        # -----------  save model -----------
        # save ModelInstance 
        model = self.model
        # save_xgboost_models(model, model_checkpoint_path)
        if not os.path.exists(model_checkpoint_path): os.makedirs(model_checkpoint_path)
        model_path = os.path.join(model_checkpoint_path, 'model.json')
        model.save_model(model_path)
        logger.info(f'Saved model to {model_checkpoint_path}')

        # save ModelInstanceArgs
        ModelInstanceArgs = self.ModelInstanceArgs  
        ModelInstanceArgs_path = os.path.join(model_checkpoint_path, 'ModelInstanceArgs.json')
        with open(ModelInstanceArgs_path, 'w') as f:
            json.dump(ModelInstanceArgs, f, indent = 4)

        # -----------  save ModelInstanceEval ----------- 
        # save ModelInstanceEval.
        eval_path = os.path.join(model_checkpoint_path, 'Eval')
        if not os.path.exists(eval_path): os.makedirs(eval_path)
        if hasattr(self, 'df_report_full'):
            eval_full_path = os.path.join(eval_path,  'eval_full.pkl')   
            df_report_full = self.df_report_full
            df_report_full.to_pickle(eval_full_path)
            eval_neat_path = os.path.join(eval_path, 'eval_neat.csv')
            df_report_neat = self.df_report_neat
            df_report_neat.to_csv(eval_neat_path, index = False)
            

    def load_model(self, model_checkpoint_path = None):
        if model_checkpoint_path is None:
            model_checkpoint_path = self.model_checkpoint_path
        model = self.model 
        model_path = os.path.join(model_checkpoint_path, 'model.json')
        model.load_model(model_path)

        self.model = model
        logger.info(f'Loaded model from {model_checkpoint_path}')

        # load ModelInstanceArgs
        data_path = os.path.join(model_checkpoint_path, 'Data')
        # OneAIDataArgs = json.load(open(os.path.join(data_path, 'OneAIDataArgs.json'), 'r'))
        OneAIDataName = os.listdir(data_path)[0]
        SPACE = self.SPACE
        self.aidata = AIData.load_aidata(data_path, OneAIDataName, SPACE)

        with open(os.path.join(model_checkpoint_path, 'ModelInstanceArgs.json'), 'r') as f:
            ModelInstanceArgs = json.load(f)
        self.ModelInstanceArgs = ModelInstanceArgs
        
        self.ModelArgs = ModelInstanceArgs['ModelArgs']
        self.TrainingArgs = ModelInstanceArgs['TrainingArgs']
        self.InferenceArgs = ModelInstanceArgs['InferenceArgs']
        self.EvaluationArgs = ModelInstanceArgs['EvaluationArgs']
        
        eval_path = os.path.join(model_checkpoint_path, 'Eval')
        eval_full_path = os.path.join(eval_path, 'eval_full.pkl')  
        if os.path.exists(eval_full_path):
            df_report_full = pd.read_pickle(eval_full_path)
            self.df_report_full = df_report_full 
        eval_neat_path = os.path.join(eval_path, 'eval_neat.csv')
        if os.path.exists(eval_neat_path):
            df_report_neat = pd.read_csv(eval_neat_path)
            self.df_report_neat = df_report_neat

    def set_aidata(self, aidata):   
        self.aidata = aidata

        
    
    def fit(self):
        aidata = self.aidata
        ModelArgs = self.ModelArgs
        TrainingArgs = self.TrainingArgs
        Name_to_Data = aidata.Name_to_Data
        TrainEvals = aidata.TrainEvals
        model = self.model
        TrainSetName = TrainEvals['TrainSetName']

        wv = self.wv 

        # model_checkpoint_path = self.model_checkpoint_path
        # if not os.path.exists(model_checkpoint_path): os.makedirs(model_checkpoint_path)
        # model_path = os.path.join(model_checkpoint_path, 'model.json')

        # if not os.path.exists(model_path):
        
        # ---- Training ----
        # model = fit_xgboost_models(model, Name_to_DsAIData, TrainSetName)
        
        Data_train = Name_to_Data[TrainSetName]
        ds_tfm = Data_train['ds_tfm']
        
        X, Y = ds_tfm['X'], ds_tfm['Y']
        
        # Update X with embeddings from items_embed_list
        items_embed_list = ['drug_name', 'npi']
        df_case = Data_train['df_case']
        for item in items_embed_list:
            if item in df_case.columns:
                name_list = df_case[item].tolist()
                embeds = wv.get_vector(name_list)
                embeds = np.array(embeds)
                embeds = embeds.reshape(len(embeds), -1)
                X = np.concatenate([X, embeds], axis=1)
        
        # Prepare evaluation sets
        eval_sets = []
        for set_name, Data in Name_to_Data.items():
            eval_X = Data['ds_tfm']['X']
            eval_Y = Data['ds_tfm']['Y']
            
            # Also update evaluation X with embeddings
            eval_df_case = Data['df_case']
            for item in items_embed_list:
                if item in eval_df_case.columns:
                    eval_name_list = eval_df_case[item].tolist()
                    eval_embeds = self.wv.get_vector(eval_name_list)
                    eval_embeds = np.array(eval_embeds)
                    eval_embeds = eval_embeds.reshape(len(eval_embeds), -1)
                    eval_X = np.concatenate([eval_X, eval_embeds], axis=1)
            
            eval_sets.append((eval_X, eval_Y))
        
        model.fit(X, Y, eval_set = eval_sets, verbose = True)

        if self.EvaluationArgs.get('show_training_visualization') == True:
            in_training_visualization(model, ModelArgs, TrainingArgs, Name_to_Data)
        # -------------------
        self.model = model


    def inference(self, Data, InferenceArgs = None):
        model = self.model
        ds_tfm = Data['ds_tfm']
        X = ds_tfm['X']

        # df_case = Data['df_case']
        # len(df_case) == len(X) 

        # items_embed_list = ['drug_name', 'npi']
        # for item in items_embed_list:
        #     assert item in df_case.columns
        #     name_list = df_case[item].tolist()
        #     wv = self.wv
        #     embeds = wv.get_vector(name_list)
        #     embeds = np.array(embeds)
        #     embeds = embeds.reshape(len(embeds), -1)
        #     X = np.concatenate([X, embeds], axis = 1)

        # concat X with wv.embeddiings. 
        # X = np.concatenate([X, drug_embeds], axis = 1)

        ####
        # concat X with wv.embeddiings. 
        # you need to find IDs. 
        # DeepFM. 
        ####

        y_pred_score = model.predict_proba(X)[:, 1] 
        
        if hasattr(self, 'df_report_full'):
            y_pred_score_percentile = self.get_inference_percentile_results(y_pred_score)
        else:
            y_pred_score_percentile = None
            
        if 'Y' in ds_tfm:
            y_real = ds_tfm['Y']
            results = {
                'y_real_label': y_real,
                'y_pred_score': y_pred_score,
                'y_pred_score_percentile': y_pred_score_percentile,
            }
        else:
            results = {
                'y_pred_score': y_pred_score,
                'y_pred_score_percentile': y_pred_score_percentile,
            }
        return results
    

    
    def evaluate(self, setname_list = None):
        # at the end of epoch and the end of training.
        aidata = self.aidata    
        EvaluationArgs = self.EvaluationArgs
        model_instance = self
        df_case_list = []
        
        if setname_list is None:
            setname_list = [aidata.TrainEvals['TrainSetName']] + aidata.TrainEvals['EvalSetNames']
        for SetName in setname_list:
            logger.info(f'Evaluate on {SetName}...')
            Data    = aidata.Name_to_Data[SetName] # Data['df_case'] (meta), Data['ds_case'] (CF). 
            # dataset = aidata.Name_to_DsAIData[SetName]  # dataset (into the model)
            df_case = Data['df_case'].copy()

            # ds_tfm = Data['ds_tfm']
            df_case['EvalName'] = SetName   
            inference_results = model_instance.inference(Data)
            for k, v in inference_results.items():
                if v is None: continue 
                print(k, len(v), len(df_case))
                df_case[k] = v
            df_case_list.append(df_case)

        df_case_eval = pd.concat(df_case_list)
        self.df_case_eval = df_case_eval


        # evalaution. 
        subgroup_config_list = EvaluationArgs['subgroup_config_list']
        y_real_label_name = EvaluationArgs['y_real_label_name']
        y_pred_score_name = EvaluationArgs['y_pred_score_name']
        EachThreshold_step = EvaluationArgs['EachThreshold_step']
        PredScoreGroup_step = EvaluationArgs['PredScoreGroup_step']
        GroupNum = EvaluationArgs['GroupNum']

        # df_case_eval_by_group = df_case_eval_by_group.copy()
        # df_case_eval_by_group = df_case_eval_by_group.dropna(subset = ['y_real_label', 'y_pred_score'])
        eval_instance = BinaryPredEval(
            df_case_eval, 
            subgroup_config_list,
            y_real_label_name,
            y_pred_score_name,
            EachThreshold_step,
            PredScoreGroup_step,
            GroupNum
        )
        self.eval_instance  = eval_instance  
        self.df_report_full = eval_instance.df_report_full
        self.df_report_neat = eval_instance.df_report_neat
        
    

    def get_inference_percentile_results(self, y_pred_score, SetName = None):
        assert hasattr(self, 'df_report_full')
        df_report_full = self.df_report_full
        
        if SetName is None:
            SetName = (self.aidata.TrainEvals['TrainSetName'], )
        
        # SetName = (SetName,)
        # print(SetName)
        report = df_report_full[df_report_full['name'] == SetName].iloc[0]
        sorted_pred_score = report['sorted_pred_score']
        y_pred_score_percentile = np.array([get_percentile(value, sorted_pred_score) for value in y_pred_score])
        return y_pred_score_percentile

