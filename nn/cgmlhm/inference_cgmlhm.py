import os 
import evaluate
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from transformers import GenerationConfig
import logging

logger = logging.getLogger(__name__)


def process_a_single_batch(model, batch, InferenceArgs = None):

    '''
    InferenceArgs = {
        'NTP_Args': {
            'num_old_tokens': 289, 
            'items_list': ['losses_each_seq', 'losses_each_token', 'predicted_ntp_labels']
        }, 
        'GEN_Args': {
            'num_old_tokens': 289,
            'max_new_tokens': 24,
            'do_sample': False,
            'items_list': ['hist', 'real', 'pred_wfe', 'logits_wfe', 'pred_nfe', 'logits_nfe'],
        },
    }
    
    batch_output = process_a_single_batch(model, batch, InferenceArgs)
    '''

    # model should be in the eval() model. 

    if InferenceArgs is None: InferenceArgs = {}

    # ------------ next-token-generation part ----------------
    NTP_Args = InferenceArgs.get('NTP_Args', None)
    if NTP_Args is not None:
        ###############################
        num_old_tokens = NTP_Args['num_old_tokens']
        items_list = NTP_Args['items_list']
        ###############################
        batch_ntp = {k: v[:, :num_old_tokens] for k, v in batch.items()}
        output = model(**batch_ntp)

        # get predicted_labels
        lm_logits = output.logits

        # get the loss each token
        labels = batch['labels'][:, :num_old_tokens]
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        logits_permuted = shift_logits.permute(0, 2, 1)
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        losses = loss_fn(logits_permuted, shift_labels)

        batch_ntp_output = {}

        if 'losses_each_seq' in items_list:
            losses_each_seq = losses.mean(dim=1).detach().cpu().numpy().tolist()
            batch_ntp_output['losses_each_seq'] = losses_each_seq

        if 'losses_each_token' in items_list:
            losses_each_token = losses.detach().cpu().numpy()
            losses_each_token = [losses_each_token[i] for i in range(len(losses_each_token))]
            batch_ntp_output['losses_each_token'] = losses_each_token

        if 'predicted_ntp_labels' in items_list:
            predicted_ntp_labels = torch.argmax(lm_logits, dim=-1)
            predicted_ntp_labels = predicted_ntp_labels.detach().cpu().numpy()# .tolist()
            predicted_ntp_labels = [predicted_ntp_labels[i] for i in range(len(predicted_ntp_labels))]
            batch_ntp_output['predicted_ntp_labels'] = predicted_ntp_labels
    else:
        batch_ntp_output = {}
        

    # ------------ generation part ----------------
    GEN_Args = InferenceArgs.get('GEN_Args', None)
    if GEN_Args is not None:
        ###############################
        items_list = GEN_Args['items_list']
        num_old_tokens = GEN_Args['num_old_tokens']
        max_new_tokens = GEN_Args['max_new_tokens']
        do_sample = GEN_Args['do_sample']
        ###############################


        HF_GenerationConfig = {}
        HF_GenerationConfig['max_new_tokens'] = max_new_tokens
        HF_GenerationConfig['do_sample'] = do_sample
        HF_GenerationConfig['return_dict_in_generate'] = True
        if any(['logits' in i for i in items_list]):
            HF_GenerationConfig['output_scores'] = True
        generation_config = GenerationConfig(**HF_GenerationConfig)

        batch_gen = {k: v[:, :num_old_tokens] for k, v in batch.items() if '--' not in k}


        # gen_outputs with future events
        if 'pred_wfe' in items_list:
            batch_gen_field_wte = {k: v for k, v in batch.items() if '--' in k}
            batch_gen_wte = {**batch_gen, **batch_gen_field_wte}
            gen_outputs_wte = model.generate(generation_config = generation_config, **batch_gen_wte)
        else:
            gen_outputs_wte = None

        # gen_outputs without future events
        if 'pred_nfe' in items_list:
            batch_gen_field_nfe = {k: v for k, v in batch.items() if '--' in k}
            for k, v in batch_gen_field_nfe.items():
                if 'event_indicators' in k:
                    v[:, num_old_tokens:] = 0   # set future events to 0    
                    batch_gen_field_nfe[k] = v
            batch_gen_nfe = {**batch_gen, **batch_gen_field_nfe}
            gen_outputs_nfe = model.generate(generation_config = generation_config, **batch_gen_nfe)
        else:
            gen_outputs_nfe = None


        
        batch_gen_output = {}
        # if 'hist' in 
        if 'hist' in items_list:
            hist = batch_gen['input_ids']
            hist = hist.cpu().numpy()
            batch_gen_output['hist'] = hist

        if 'real' in items_list:
            real = batch['labels'][:, num_old_tokens: num_old_tokens+max_new_tokens]
            real = real.cpu().numpy()
            batch_gen_output['real'] = real

        if 'pred_wfe' in items_list:
            sequences = gen_outputs_wte['sequences']
            pred_wfe = sequences[:, -max_new_tokens:]
            pred_wfe = pred_wfe.cpu().numpy()
            batch_gen_output['pred_wfe'] = pred_wfe

        if 'logits_wfe' in items_list:
            logits_wfe = gen_outputs_wte['scores']
            logits_wfe = np.array([logit.cpu().numpy() 
                                    for logit in logits_wfe]
                                    ).transpose(1, 0, 2) 
            batch_gen_output['logits_wfe'] = logits_wfe

        if 'pred_nfe' in items_list:
            sequences = gen_outputs_nfe['sequences']
            pred_nfe = sequences[:, -max_new_tokens:]
            pred_nfe = pred_nfe.cpu().numpy()
            batch_gen_output['pred_nfe'] = pred_nfe

        if 'logits_nfe' in items_list:
            logits_nfe = gen_outputs_nfe['scores']
            logits_nfe = np.array([logit.cpu().numpy() 
                                    for logit in logits_nfe]
                                    ).transpose(1, 0, 2) 
            batch_gen_output['logits_nfe'] = logits_nfe


        batch_gen_output = {
            k: [v[i] for i in range(v.shape[0])] for k, v in batch_gen_output.items()
        }
    else:
        batch_gen_output = {}

    batch_output = {**batch_ntp_output, **batch_gen_output}
    return batch_output


def inference_model_with_ds(model, ds_tfm, InferenceArgs, folder = None):
    max_inference_num = InferenceArgs.get('max_inference_num', None)
    save_df = InferenceArgs.get('save_df', False)
    load_df = InferenceArgs.get('load_df', False)
    chunk_size = InferenceArgs.get('chunk_size', 12800)
    batch_size = InferenceArgs.get('batch_size', 64)
    if folder is None:
        save_df = False; load_df = False; folder = ''

    # ds_tfm = Data['ds_tfm']
    # df_case = Data['df_case']
    if max_inference_num is not None: 
        ds_tfm = ds_tfm.select(range(max_inference_num))
        # df_case = df_case.iloc[:max_inference_num]

    columns = ds_tfm.column_names
    columns_tag = [i for i in columns if '--' not in i]
    df_case = ds_tfm.select_columns(columns_tag).to_pandas()

    # df_case = ds_tfm.select_columns(['CaseID', 'ObsDT', 'cf.TargetCGM_Bf24H', 'cf.TargetCGM_Af2H'])
    chunk_numbers = len(df_case) // chunk_size

    li_df_eval_chunk_path = []
    li_df_eval_chunk_data = []

    for chunk_id in range(chunk_numbers + 1):
        start = chunk_id * chunk_size
        end = min((chunk_id+1) * chunk_size, len(df_case))
        
        df_case_chunk = df_case.iloc[start:end].reset_index(drop = True)
        ds_tfm_chunk = ds_tfm.select(range(start, end))
        
        # folder = os.path.join(self.model_artifact_path, 'Inference')
        file = os.path.join(folder, f'chunk_{chunk_id:06}_s{start}_e{end}.p')
        
        if save_df == True and not os.path.exists(folder):
            os.makedirs(folder)
            logger.info(f'Creating folder {folder}')

        if load_df == True and os.path.exists(file):
            logger.info(f'Loading chunk {chunk_id} from {file}')
            li_df_eval_chunk_path.append(file)
            continue
        
        df_eval_chunk = pd.DataFrame()
        if len(ds_tfm_chunk) <= 10:
            iterrator_object = range(0, len(ds_tfm_chunk), batch_size)
        else:
            iterrator_object = tqdm(range(0, len(ds_tfm_chunk), batch_size))
            
        for batch_s in iterrator_object:
            batch_e = min(batch_s + batch_size, len(ds_tfm_chunk))
            batch = ds_tfm_chunk[batch_s: batch_e]
            for k, v in batch.items():
                batch[k] = v.to(model.device)
            with torch.no_grad():
                model.eval()
                # logger.info(f'Processing chunk {chunk_id} batch {batch_s} to {batch_e}: {batch}')
                output = process_a_single_batch(model, batch, InferenceArgs)
                # logger.info(f'Output: {output}')

            df_batch = pd.DataFrame(output)
            # logger.info(df_batch)
            df_eval_chunk = pd.concat([df_eval_chunk, df_batch], axis = 0)
            # logger.info(df_eval_chunk)
        
        df_eval_chunk = df_eval_chunk.reset_index(drop=True)

        # df_chunk_caseids = dataset_chunk.select_columns(case_id_columns).to_pandas()
        assert len(df_eval_chunk) == len(df_case_chunk), f'{len(df_eval_chunk)} != {len(df_case_chunk)}'
        df_chunk = pd.concat([df_case_chunk, df_eval_chunk], axis = 1)
        df_chunk = df_chunk.reset_index(drop=True)

        if save_df == True:
            logger.info(f'Saving chunk {chunk_id} to {file}')
            df_chunk.to_pickle(file)
            li_df_eval_chunk_path.append(file)
        else:
            li_df_eval_chunk_data.append(df_chunk)

    if save_df == True:
        df_case_eval = pd.concat([pd.read_pickle(file) for file in li_df_eval_chunk_path], axis = 0)
    else:
        df_case_eval = pd.concat(li_df_eval_chunk_data, axis = 0)
    
    results = {
        'df_case_eval': df_case_eval,
    }
    return results


def compute_metrics_for_ntp(eval_preds, experiment_id, AfTknNum = 24):

    metric_acc = evaluate.load("accuracy", experiment_id = experiment_id)
    metric_mse = evaluate.load('mse',      experiment_id = experiment_id)

    preds, labels = eval_preds
    # print(preds.shape, labels.shape)
    # print(preds.shape, labels.shape)
    # preds have the same shape as the labels, after the argmax(-1) has been calculated
    # by preprocess_logits_for_metrics but we need to shift the labels
    labels = labels[:, 1:]
    preds  = preds[:, :-1] 
    # print(preds.shape, labels.shape)

    all_labels = labels.reshape(-1)
    all_preds = preds.reshape(-1)
    # print(all_labels.shape, all_preds.shape)


    af_labels = labels[:, -AfTknNum:].reshape(-1)
    af_preds  = preds[:, -AfTknNum:].reshape(-1)
    # print(af_labels.shape, af_preds.shape)
    
    d_accu = metric_acc.compute(predictions=all_preds, references=all_labels)
    d_mse = metric_mse.compute(predictions=all_preds, references=all_labels)
    d_accu_af = metric_acc.compute(predictions=af_preds, references=af_labels)
    d_mse_af = metric_mse.compute(predictions=af_preds, references=af_labels)
    
    d = {}
    for k, v in d_accu.items(): d[k] = v
    for k, v in d_accu_af.items(): d[k + '_af'] = v

    for k, v in d_mse.items(): d[k] = v
    for k, v in d_mse_af.items(): d[k + '_af'] = v

    d['rMSE'] = np.sqrt(d['mse'])
    d['rMSEaf'] = np.sqrt(d['mse_af'])
    
    d['ACUU']   = d['accuracy'] # np.sqrt()
    d['ACUUaf'] = d['accuracy_af'] # np.sqrt()
    
    del d['mse'], d['mse_af'], d['accuracy'], d['accuracy_af']
    return d



    
def compute_metrics_generation(eval_preds, experiment_id):
    metric_acc = evaluate.load("accuracy", experiment_id = experiment_id)
    metric_mse = evaluate.load('mse', experiment_id = experiment_id)

    preds, labels = eval_preds
    print(preds.shape, labels.shape)
    preds = preds[:, -labels.shape[-1]:]
    print(preds.shape, labels.shape)

    d = {}
    for pred_horizon in [6, 12, 24]:
        d_accu = metric_acc.compute(predictions=preds[:, :pred_horizon].reshape(-1), 
                                    references=labels[:, :pred_horizon].reshape(-1))
        d_mse = metric_mse.compute(predictions=preds[:, :pred_horizon].reshape(-1),  
                                   references=labels[:, :pred_horizon].reshape(-1))
        
        for k, v in d_accu.items(): d[k + f'_ph{pred_horizon}'] = v
        for k, v in d_mse.items():  d[k + f'_ph{pred_horizon}'] = v
        d['rmse'+ f'_ph{pred_horizon}'] = np.sqrt(d['mse'+ f'_ph{pred_horizon}'])
    return d

