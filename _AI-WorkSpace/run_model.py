
import logging
import math
import os
import sys
import warnings
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional
import random
import datasets
import evaluate
import torch
from datasets import load_dataset
import tokenizers
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import time 
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    is_torch_tpu_available,
    set_seed,
)

from transformers import Trainer, TrainingArguments, TrainerCallback
from datetime import datetime

from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from datasets import disable_caching


disable_caching()


random.seed(42)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.37.0.dev0")

# require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

class TimestampCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        # Add the current timestamp to the logs
        logs["step"] = state.global_step
        logs["timestamp"] = str(datetime.now())
        

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    check_dataset_only: Optional[str] = field(
        default=False, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")

        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


# ------------------------------- part 1: model_args -------------------------------
def prepare_parser_and_args(ModelArguments, DataTrainingArguments, TrainingArguments):
    # ------------------------------- part 1: model_args -------------------------------
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead.",
            FutureWarning,
        )
        if model_args.token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        model_args.token = model_args.use_auth_token

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clm", model_args, data_args)
    return parser, model_args, data_args, training_args


# ------------------------------- part 2: logging -------------------------------
def prepare_logging(training_args):
    # ------------------------------- part 2: logging -------------------------------
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    return logger

# ------------------------------- part 3: last checkpoint -------------------------------
def prepare_lastcheckpoint(training_args):
    # ------------------------------- part 3: last checkpoint -------------------------------
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    return last_checkpoint


# ------------------------------- part 5: dataset -------------------------------
def prepare_raw_datasets(groupname_types, 
                         scope_type, 
                         CaseFolder,
                         CaseObsFolder,
                         streaming = False,
                         output_dir = None, logger = None):
    
    total_groupname_list = sorted([i.split('.')[0] for i in os.listdir(CaseFolder) if '.p' in i and scope_type in i])
    # total_groupname_list = {k: v for k, v in enumerate(total_groupname_list)}
    total_groupname_list = {int(v.split('_')[0]): v for v in total_groupname_list}

    groupname_dict = total_groupname_list.copy()
    for groupname_type in groupname_types:
        groupname_dict = {k: v for k, v in groupname_dict.items() if groupname_type in v}

    groupname_ids = [i for i in groupname_dict]
    groupname_selected_list = [groupname_dict[i] for i in groupname_ids]

    L = []
    groupname_to_ds = {}
    for idx, groupname in enumerate(groupname_selected_list):
        d = {}
        d['group_id'] = groupname_ids[idx]
        d['groupname'] = groupname
        group_id = groupname_ids[idx]

        # -------------- Part 1: aicasetkn_file
        aicase_file = os.path.join(HfAICase_Folder, groupname + '_info.p')
        if not os.path.exists(aicase_file):
            print('><', group_id, groupname, aicase_file, '<---------------- does not exist. pass it.')
            continue
        df_case = pd.read_pickle(aicase_file)
        d['aicase_num'] = df_case.shape[0]
        L.append(d)
        
        # -------------- Part 2: aicasetknidx_file

        aicasetknidx_file = os.path.join(CaseObsFolder, f'{groupname}_tkn_{len(df_case)}')
        ds = datasets.load_from_disk(aicasetknidx_file)
        groupname_to_ds[groupname] = ds
    
    df = pd.DataFrame(L)
    if len(df) == 0: 
        print('>< No AICase in the dataset. <----------------')
        logger.info(f"No AICase in the dataset. ")
    else:
        total_aicase_num = df['aicase_num'].sum()
        print(f"Total Cases Number in the raw dataset: {total_aicase_num:,}")
        logger.info(f"Total Cases Number in the raw dataset: {total_aicase_num:,}")
        df['aicase_num'] = df['aicase_num'].apply(lambda x: f"{x:,}")
    print(df)

    if output_dir is not None:
        datainfo_path = os.path.join(output_dir, 'datainfo.csv')
        os.makedirs(os.path.dirname(datainfo_path), exist_ok=True)
        print(f"Save data info to: {datainfo_path}. ")
        logger.info(f"Save data info to: {datainfo_path}. ")
        df.to_csv(datainfo_path, index=False)

    ds_tknidx = datasets.concatenate_datasets([groupname_to_ds[i] for i in groupname_to_ds])
    print(ds_tknidx)
    
    train_test_split = ds_tknidx.train_test_split(test_size=0.0001, seed=42)
    
    dataset_to_size = {
        'train': len(train_test_split['train']),
        'validation': len(train_test_split['test']),
    }
    
    if streaming == False:
        raw_datasets = datasets.DatasetDict({
            'train': train_test_split['train'],
            'validation': train_test_split['test']
        })
    else:
        # ds_tknidx = ds_tknidx.to_iterable_dataset()
        raw_datasets = datasets.IterableDatasetDict({
            'train': train_test_split['train'].to_iterable_dataset(),
            'validation': train_test_split['test'].to_iterable_dataset(),
        })
    return raw_datasets, dataset_to_size


# ------------------------------- part 6: load pretrained model tokenizer -------------------------------
def load_pretrained_tokenizer(model_args):
    # ------------------------------- part 7: load pretrained model tokenizer -------------------------------
    tokenizer_path = model_args.tokenizer_name
    tokenizer = tokenizers.Tokenizer.from_file(tokenizer_path)
    return tokenizer 


# ------------------------------- part 7: load pretrained model config -------------------------------
def load_pretrained_model_config(model_args, tokenizer):
    # ------------------------------- part 6: load pretrained model config -------------------------------
    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
        ###########
        'vocab_size': tokenizer.get_vocab_size(),
        ###########
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")
    return config


# ------------------------------- part 8: load pretrained model -------------------------------
def load_pretrained_model(model_args, config, tokenizer):
    # ------------------------------- part 8: load pretrained model -------------------------------
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=model_args.trust_remote_code)
    n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
    logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")
    return model 


# ------------------------------- part 11: process tokenizedidx lm datasets. -------------------------------
def get_lm_datasets_to_tensor(raw_datasets, training_args, data_args, model_args, config, tokenizer):
    # or your get the tensor lm datasets.
    # ---------------------------- part 11.3: transform function -------------------------------
    def transform_clm_result(examples):
        # Split by chunks of max_len.
        # print([len(i) for i in examples['input_ids']])
        # input_ids = torch.LongTensor(examples['input_ids'])
        # print(input_ids.shape)
        result = {
            'input_ids': examples['input_ids'], # torch.LongTensor(examples['input_ids']), # torch.LongTensor(examples['input_ids']), 
            'labels': examples['input_ids'], # torch.LongTensor(examples['input_ids']), # torch.LongTensor(examples['input_ids'])
        }
        return result

    with training_args.main_process_first(desc="get clm data"):
        # here we have two types of datasets.
        if not data_args.streaming:
            # either memory-mapped dataset
            # you have a new dataset now.
            # lm_datasets = raw_datasets.map(
            #     transform_clm_result,
            #     batched=True,
            #     num_proc=data_args.preprocessing_num_workers,
            #     load_from_cache_file=not data_args.overwrite_cache,
            #     desc=f"Process data into clm data input",
            # )
            raw_datasets.set_transform(
                transform_clm_result,
                # batched=True,
                # num_proc=data_args.preprocessing_num_workers,
                # load_from_cache_file=not data_args.overwrite_cache,
                # desc=f"Process data into clm data input",
            )
            lm_datasets = raw_datasets
            # lm_datasets.remove_columns(['PID', 'PredDT'])
        else:
            # streaming dataset
            lm_datasets = raw_datasets.map(
                transform_clm_result,
                batched=True,
            )
    lm_datasets = lm_datasets.remove_columns(['PID', 'PredDT'])
    return lm_datasets


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    # ------------------------------- part 1: model_args -------------------------------
    # parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    # model_args, data_args, training_args = parser.parse_args_into_dataclasses(args)
    parser, model_args, data_args, training_args = prepare_parser_and_args(ModelArguments, DataTrainingArguments, TrainingArguments)



    # ------------------------------- part 2: logging -------------------------------
    # Setup logging
    logger = prepare_logging(training_args)

    # ------------------------------- part 3: last checkpoint -------------------------------
    # Detecting last checkpoint.
    last_checkpoint = prepare_lastcheckpoint(training_args)
    print(f'\n-----> last_checkpoint: {last_checkpoint}\n')

    # ------------------------------- part 4: random_seed -------------------------------
    # Set seed before initializing model.
    set_seed(training_args.seed)


    #  ------------------------------- part 5: dataset -------------------------------
    # raw_datasets = prepare_raw_datasets(data_args, model_args)

    DATASPACE_TASK = '4-Data_CGMGPT'
    groupname_types = data_args.dataset_name.split('&')
    CaseFolder = os.path.join(DATASPACE_TASK, 'CaseFolder') # '4-Data_CGMGPT/CaseFolder/'
    # HfScopeInfo_Folder = os.path.join(DATASPACE_TASK, 'HfScopeCase')
    HfAICase_Folder = os.path.join(DATASPACE_TASK, 'HfAICase')
    HfAICaseTknIdx_Folder = os.path.join(DATASPACE_TASK, 'HfAICaseTknIdx')

    raw_datasets, dataset_to_size = prepare_raw_datasets(groupname_types, 
                                                        CaseFolder,
                                                        HfAICase_Folder, 
                                                        HfAICaseTknIdx_Folder, 
                                                        streaming = data_args.streaming,
                                                        logger=logger)
    

    # ------------------------------- part 7: load pretrained model tokenizer -------------------------------
    tokenizer = load_pretrained_tokenizer(model_args)

    
    # ------------------------------- part 6: load pretrained model config -------------------------------
    config = load_pretrained_model_config(model_args, tokenizer)
    
    # ------------------------------- part 8: load pretrained model -------------------------------
    model = load_pretrained_model(model_args, config, tokenizer) 

    # ------------------------------- part 9: Preprocessing the datasets. -------------------------------
    lm_datasets = get_lm_datasets_to_tensor(raw_datasets, training_args, data_args, model_args, config, tokenizer)

    # ---------------------------- part A: get train_dataset ----------------------------
    train_dataset = lm_datasets["train"]
    if data_args.check_dataset_only:
        for i in range(0, len(train_dataset), training_args.per_device_train_batch_size):
            batch = train_dataset[i:i+training_args.per_device_train_batch_size]
            # print(batch)
            x = batch['input_ids']
            y = batch['labels']
            x = np.array(x)
            y = np.array(y)
            if i == 0: 
                print(x.shape)
                shape = x.shape

            if not shape == x.shape:
                print('><', i, shape, x.shape, y.shape)
            if i % 100000 == 0: 
                print(i, shape, x.shape, y.shape)
        exit(0)
        
    # ---------------------------- part B: get eval_dataset ----------------------------
    if training_args.do_eval:
        eval_dataset = lm_datasets["validation"]
 
        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        metric = evaluate.load("accuracy")

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            return metric.compute(predictions=preds, references=labels)

    # ---------------------------- part C: Initialize our Trainer ----------------------------
    if data_args.streaming:
        train_dataset_samples = dataset_to_size['train']
        print('---- train_dataset_samples:', train_dataset_samples)
        num_batches_per_epoch = train_dataset_samples // (training_args.per_device_train_batch_size * training_args._n_gpu)
        print('---- num_batches_per_epoch:', num_batches_per_epoch)

        # print('---- num_train_epochs:', training_args.num_train_epochs)

        max_steps = int(training_args.num_train_epochs * num_batches_per_epoch)
        print('---- max_steps:', max_steps)
        training_args.max_steps = max_steps
        
        training_args.num_train_epochs = int(training_args.num_train_epochs)
        print('---- num_train_epochs:', training_args.num_train_epochs)

    # this part is so important. 
    trainer = Trainer(

        ########## you have your model 
        model = model,
        ########## you have your training_args
        args=training_args,
        ########## get train_dataset
        train_dataset=train_dataset, # if training_args.do_train else None,
        ########## get eval_dataset
        eval_dataset=eval_dataset, #  if training_args.do_eval else None,
        ########## huge question here: is it ok to ignore the tokenizer?
        # tokenizer=tokenizer,
        ########## huge question here: data_collator
        data_collator=default_data_collator,
        
        compute_metrics=compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval and not is_torch_tpu_available()
        else None,

        callbacks=[TimestampCallback],
    )

    # ---------------------------- part D: Do Train ----------------------------
    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint


        
        ###########################################################
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        ###########################################################

        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # ---------------------------- part E: Do Evaluation ----------------------------
    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        ###########################################################
        metrics = trainer.evaluate()
        ###########################################################

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
