import os
import json
import numpy as np
# import xgboost as xgb
import matplotlib.pyplot as plt
from datetime import datetime 
from datasets.fingerprint import Hasher
# from recfldtkn.model_base.model_instance import ModelInstance
import logging 
import torch
import pandas as pd
from .training_bandit import train_bandit_model, sample_action_batch, my_softmax, mean0, var0
from .DROPO import regression_utility as ru
from nn.eval.banditpred import BanditPredEval
import numpy as np
import os
# from pickle import dump, load
import pickle 
import json 
from sklearn.preprocessing import StandardScaler
# from pickle import dump, load
# import pickle
import copy 
    
from recfldtkn.aidata_base.aidata import AIData


logger = logging.getLogger(__name__)

class train_args(object):
    def __init__(self, **kwargs):
        self.batch_size = 256
        self.epochs_training = 20
        self.lr = 0.001
        self.lr_pg = 0.0005
        self.momentum = 0.5
        self.clip_weight = True
        self.weights_upper_bound = 100
        self.weights_lower_bound = 0.01
        self.epochs_policy_gradient = 10
        self.lr_decay = 0.9
        self.evaluate_batch = 200
        self.seed = 10
        self.set_seed = False
        
        for k, v in kwargs.items():
            setattr(self, k, v)


        
    def __getitem__(self,item):
        return getattr(self, item)
    

class BanditInstance:
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
        self.aidata = aidata
        if self.aidata is not None:
            self.OneEntryArgs = copy.deepcopy(aidata.OneEntryArgs)
            self.Name_to_Data = aidata.Name_to_Data


        self.ModelArgs = ModelArgs
        if TrainingArgs is None:
            self.TrainingArgs = train_args()
  
        else:
            self.TrainingArgs = TrainingArgs
        

        self.InferenceArgs = InferenceArgs
        self.EvaluationArgs = EvaluationArgs
        self.SPACE = SPACE
        
        if type(self.TrainingArgs) == dict:
            self.TrainingArgs = train_args(**self.TrainingArgs)
            
        self.ModelInstanceArgs = {
            'ModelArgs': self.ModelArgs,
            'TrainingArgs': self.TrainingArgs.__dict__,
            'InferenceArgs': self.InferenceArgs,
            'EvaluationArgs': self.EvaluationArgs,
            'SPACE': SPACE,
        }

        self.ModelNames = ModelNames
        self.model_checkpoint_name = self.ModelNames['model_checkpoint_name']
        self.MODEL_ENDPOINT = self.ModelNames['MODEL_ENDPOINT']
        self.model_checkpoint_path = os.path.join(SPACE['MODEL_ROOT'], 
                                                  self.MODEL_ENDPOINT, 
                                                  'models', 
                                                  self.model_checkpoint_name, 
                                                  )
        
        
    def init_model(self):
        pass
        # ModelArgs = self.ModelArgs
        # args = {k: v for k, v in ModelArgs.items() if k != 'algorithm'}
        # model = xgb.XGBClassifier(**args, **self.TrainingArgs)
        # self.model = model 
    
    def save_model(self, model_checkpoint_path = None):
        model_dict = self.model_dict
        
        if model_checkpoint_path == None:
            model_checkpoint_path = self.model_checkpoint_path

        # # -----------  save aidata -----------
        data_path = os.path.join(model_checkpoint_path, 'Data')
        if not os.path.exists(data_path): os.makedirs(data_path)
        aidata = self.aidata
        logger.info(f'Save model to {model_checkpoint_path}')
        aidata.save_aidata(data_path, save_args = {'sample_num': 10})
        #
        # logging_policy_path = os.path.join(model_checkpoint_path, 'LoggingPolicy.npy')
        # np.save(logging_policy_path, self.logging_policy)

        # -----------  save model -----------
        model = model_dict['model']
        scaler =  model_dict['scaler']
        model_robust_list = model_dict['model_robust_list']


        if not os.path.exists(model_checkpoint_path): os.makedirs(model_checkpoint_path)
        model_path = os.path.join(model_checkpoint_path, 'model.pth')
        torch.save(model,model_path)
        
        for i in range(len(model_robust_list)):
            model_path = os.path.join(model_checkpoint_path, 'reward' + str(i)+'.json')
            model_robust_list[i].save_model(model_path)

        scaler_path = os.path.join(model_checkpoint_path, 'scaler.pkl')
        pickle.dump(scaler, open(scaler_path, 'wb'))

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
        
        # load ModelInstanceArgs
        # print(model_checkpoint_path)
        data_path = os.path.join(model_checkpoint_path, 'Data')
        # OneAIDataArgs = json.load(open(os.path.join(data_path, 'OneAIDataArgs.json'), 'r'))
        # OneAIDataName = OneAIDataArgs['OneAIDataName']
        OneAIDataName = os.listdir(data_path)[0]
        SPACE = self.SPACE
        aidata = AIData.load_aidata(data_path, OneAIDataName, SPACE)
        self.aidata = aidata
        if self.aidata is not None:
            self.OneEntryArgs = copy.deepcopy(aidata.OneEntryArgs)
        
        # logging_policy_path = os.path.join(model_checkpoint_path, 'LoggingPolicy.npy')
        # self.logging_policy = np.load(logging_policy_path)


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

        # ---load model---
        model = torch.load(os.path.join(model_checkpoint_path, 'model.pth'), weights_only=False)

        model_robust_list = []
        filename_list = sorted(os.listdir(model_checkpoint_path))
        filename_list = [i for i in filename_list if '._' not in i]

        for i in range(len(filename_list)):
            if 'reward' in filename_list[i]:
                reward_model_path = os.path.join(model_checkpoint_path, filename_list[i])
                import xgboost as xgb
                reward_model = xgb.XGBRegressor()  # Create an instance
                reward_model.load_model(reward_model_path)  # Load model in-place
                logger.info(f'Load reward model from {reward_model_path} and get model: {reward_model}')
                model_robust_list.append(reward_model)

        
        scaler_checkpoint = os.path.join(model_checkpoint_path, 'scaler.pkl')
        scaler = StandardScaler()
        scaler = pickle.load(open(scaler_checkpoint, 'rb'))
        model_dict = {}
        model_dict['model'] = model
        model_dict['scaler'] = scaler
        model_dict['model_robust_list'] = model_robust_list
        self.model_dict = model_dict
        self.model = model
        return model_dict

    
    def fit(self):
        # train_set = self.aidata.Name_to_DsAIData['Train']
        # test_set = self.aidata.Name_to_DsAIData['Test']



        X = self.Name_to_Data['In-Train']['ds_tfm']['X'].toarray()
        A = self.Name_to_Data['In-Train']['ds_tfm']['A']
        R = self.Name_to_Data['In-Train']['ds_tfm']['Y']

        X = np.nan_to_num(X, nan=0.0)


        test_X = self.Name_to_Data['In-Test']['ds_tfm']['X'].toarray()
        test_A = self.Name_to_Data['In-Test']['ds_tfm']['A']
        test_R = self.Name_to_Data['In-Test']['ds_tfm']['Y']
        test_X = np.nan_to_num(test_X, nan=0.0)



        # self.model_checkpoint_path = '/Users/yihomh/Downloads/dfirst-ai-space-sms-bandit/Model'

        # print(self.model_checkpoint_path)
        model_dict, eval_loss = train_bandit_model(
            self.ModelArgs, 
            self.TrainingArgs, 
            self.model_checkpoint_path,
            X, A, R, test_X, test_A, test_R
        )
        # self.logging_policy = logging_policy
        self.model_dict = model_dict
        self.eval_loss = eval_loss
        self.model = model_dict['model']
        # self.save_model(self.model_checkpoint_path)



    def inference_bandit(self, x, model_dict):
        # print(logging_policy.shape)
        x = torch.tensor(model_dict['scaler'].transform(x))
        # print('x', x.shape) 

        prob_all = model_dict['model'](x).detach()
        # print('prob_all', prob_all.shape)
        # print(prob_all)
        # prob_all = self.set_action_to_0(prob_all)
        prob_all = my_softmax(prob_all)
        # print('prob_all', prob_all.shape)
        # print(prob_all)
        action = sample_action_batch(prob_all)
        # print('action:', action.shape)
        prob = prob_all[np.arange(len(prob_all)), action]
        # print('prob', prob.shape)
        # weight = torch.tensor(logging_policy[action]/prob)

        reward_training = np.ones(len(prob))
        # print('reward_training:', reward_training.shape)

        for h in range(prob_all.shape[1]):

            # do the batch reward.
            index = action==h # (1,) # (1,1)
            # print('=========')
            # print('h:', h)
            # print('x:', x.shape)
            # print('index:', index.shape, index)
            features = x[index]
            # print('features:', features.shape, features)

            if len(features) == 0:
                # in this batch, these is no datapoints for this action
                # print(f'no datapoints for this action: {h}')
                continue 
            # weight_index = weight[index]
            # act = action[index]
            output = model_dict['model_robust_list'][h].predict(features)
            # print('output:', output.shape, output)
            # meanY_robust, varY = ru.predict_regression(torch.tensor(weight_index), model_dict['Myy_robust_list'][h], model_dict['Myx_robust_list'][h], output, mean0, var0)
            reward_training[index] = output
        return action, reward_training, prob_all

    def inference(self, Data, InferenceArgs = None):
        # X = dataset['X'].toarray()
        action_to_id = self.OneEntryArgs['Output_Part']['action_to_id']
        id_to_action = {v: k for k, v in action_to_id.items()}

        ds_tfm = Data['ds_tfm']
        X = ds_tfm['X'].toarray()
        action, predicted_reward, action_prob = self.inference_bandit(X, self.model_dict)
        action_name_list = [id_to_action[i] for i in action]
        results = {
            'Action':action,
            'ActionName':action_name_list,
            'Reward':predicted_reward, # [0.47142]
            # 'action_prob': action_prob,
        }
        
        for i in range(action_prob.shape[1]):
            action_name = id_to_action[i]
            results['action_prob_' + action_name] = action_prob[:,i].cpu().numpy()
        # print(np.sum(action!=0))
        #
        # for i in range(len(action_prob[0])):
        #     results['Action Probability ' + str(i)] = np.array(action_prob[:,i])

        return results
    
    def evaluate(self):
        Name_to_Data = self.Name_to_Data
        InferenceArgs = self.InferenceArgs
        # aidata = self.aidata
        EvaluationArgs = self.EvaluationArgs
        model_instance = self
        df_case_list = []
        SetName_list = [i for i in Name_to_Data.keys() if i != 'train']
        for SetName in SetName_list:
            logger.info(f'Evaluate on {SetName}...')
            Data = Name_to_Data[SetName]
            df_case = Data['df_case'].copy()
            df_case['EvalName'] = SetName
            inference_results = model_instance.inference(Data, InferenceArgs)
            for k, v in inference_results.items():
                # print(k, len(v), len(Data['ds_tfm']['X'].toarray()))
                # print(k)
                df_case[k] = v
            df_case_list.append(df_case)


        df_case_eval = pd.concat(df_case_list)
        self.df_case_eval = df_case_eval

        subgroup_config_list = EvaluationArgs['subgroup_config_list']
        action_name = EvaluationArgs['action_name']
        reward_predicted_name = EvaluationArgs['reward_predicted_name']
        action_distribution_name = EvaluationArgs['action_distribution_name']


        # df_case_eval_by_group = df_case_eval_by_group.copy()
        # df_case_eval_by_group = df_case_eval_by_group.dropna(subset = ['y_real_label', 'y_pred_score'])
        eval_instance = BanditPredEval(
            df_case_eval,
            subgroup_config_list,
            action_name,
            reward_predicted_name,
            action_distribution_name
        )
        self.eval_instance  = eval_instance
        self.df_report_full = eval_instance.df_report_full
        self.df_report_neat = eval_instance.df_report_neat

