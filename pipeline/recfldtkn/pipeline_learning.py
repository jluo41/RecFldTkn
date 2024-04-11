import os 
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s:%(asctime)s:(%(filename)s@%(lineno)d %(name)s)]: %(message)s')


def train_machine_learning_model(model_args, 
                                 training_args, 
                                 model_checkpoint_path,
                                 ds_case_final_dict,
                                 TrainSetName, 
                                 EvalSetNames,
                                 fn_save_model, 
                                 fn_load_model,
                                 LOAD_MODEL = False,
                                 SAVE_MODEL = True,
                                 ):
    
    algorithm = model_args['algorithm']
    args = {k: v for k, v in model_args.items() if k != 'algorithm'}

    if algorithm == 'XGBClassifier':
        from xgboost import XGBClassifier 
        model = XGBClassifier(**args, **training_args)

    elif algorithm == 'AdaBoostClassifier':
        from sklearn.ensemble import AdaBoostClassifier
        model = AdaBoostClassifier(**args, **training_args)

    elif algorithm == 'LGBMClassifier':
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(**args, **training_args)

    elif algorithm == 'LogisticRegression':
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()

    elif algorithm == 'RandomForestClassifier':
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(**args)

    elif algorithm == 'SVM':
        from sklearn.svm import SVC
        model = SVC(**args,**training_args)
    else:
        raise ValueError(f'not yet support {algorithm}')


    if not os.path.exists(model_checkpoint_path) or LOAD_MODEL == False:
        train_set = ds_case_final_dict[TrainSetName]
        eval_sets = [(eval_set['X'], eval_set['Y']) for k, eval_set in ds_case_final_dict.items()]
        set_names = [k for k in ds_case_final_dict]
        X, Y = train_set['X'], train_set['Y']
        try:
            model.fit(X, Y, eval_set = eval_sets, verbose = True)

            import matplotlib.pyplot as plt
            evals_result = model.evals_result()
            results = [v for k, v in evals_result.items()]
            evals_result = dict(zip(set_names, results))
            print([i for i in evals_result])
            eval_metric = training_args['eval_metric']
            epochs = len(evals_result[set_names[0]][eval_metric])
            x_axis = range(0, epochs)
            fig, ax = plt.subplots()
            for splitname in evals_result:
                ax.plot(x_axis, evals_result[splitname]['logloss'], label=splitname)
            ax.legend()
            plt.ylabel('Log Loss')
            plt.title(f'{algorithm} Log Loss')
            plt.show()
        except:
            model.fit(X, Y)

    else:
        logger.info(f'loading model from {model_checkpoint_path}')
        model = fn_load_model(model_checkpoint_path) 

    if SAVE_MODEL == True:
        if not os.path.exists(model_checkpoint_path): 
            os.makedirs(model_checkpoint_path)

        logger.info(f'save model to {model_checkpoint_path}')
        fn_save_model(model, model_checkpoint_path)

    return model
