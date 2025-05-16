
from collections import OrderedDict
from typing import Any, List, Mapping, Optional

from transformers import PreTrainedTokenizer, TensorType, is_torch_available
from transformers.configuration_utils import PretrainedConfig
from transformers.onnx import OnnxConfigWithPast, PatchingSpec
from transformers.utils import logging


logger = logging.get_logger(__name__)

# from nn.cgmlhm.configuration_cgmlhm import CgmLhmConfig


from collections import OrderedDict
from typing import Any, List, Mapping, Optional

from transformers import PreTrainedTokenizer, TensorType, is_torch_available
from transformers.configuration_utils import PretrainedConfig
from transformers.onnx import OnnxConfigWithPast, PatchingSpec
from transformers.utils import logging

logger = logging.get_logger(__name__)


class CgmLhmConfig(PretrainedConfig):
    model_type = "cgmlhm"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        'n_layer': 'tf_n_layer',
        "hidden_size": "n_embd",
        "max_position_embeddings": "n_positions",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
        "layer_norm_epsilon": "layer_norm_eps",
        "hidden_act": "activation_function",
        # "hidden_dropout_prob": 
    }

    def __init__(
        self,
        n_embd=768,
        initializer_range=0.02,
        use_cache=True,
        early_fusor_strategy = True, 
        late_fusor_strategy = False, 

        time_field = 'Time',

        # lsm_config
        lsm_n_positions=1024,
        lsm_n_layer=12,
        lsm_n_head=12,
        lsm_n_inner=None,
        lsm_activation_function="gelu_new",
        lsm_resid_pdrop=0.1,
        lsm_embd_pdrop=0.1,
        lsm_attn_pdrop=0.1,
        lsm_layer_norm_epsilon=1e-5,
        # lsm_initializer_range=0.02,
        lsm_summary_type="cls_index",
        lsm_summary_use_proj=True,
        lsm_summary_activation=None,
        lsm_summary_proj_to_labels=True,
        lsm_summary_first_dropout=0.1,
        lsm_scale_attn_weights=True,
        lsm_scale_attn_by_inverse_layer_idx=False,
        lsm_reorder_and_upcast_attn=False,


        # fieldeconder_settings.
        fe_num_hidden_layers=6,
        fe_timestep_lookback = 600,
        fe_timestep_lookahead = 300,
        fe_embd_pdrop = 0.1,
        fe_use_field_type_embedding = True,
        fe_num_attention_heads=12,
        fe_intermediate_size=3072,
        fe_hidden_act="gelu",
        fe_hidden_dropout_prob=0.1,
        fe_attention_probs_dropout_prob=0.1,
        fe_max_position_embeddings=512,
        # fe_initializer_range=0.02,
        fe_layer_norm_eps=1e-12,
        fe_position_embedding_type="absolute",
        # fe_use_cache=True,
        fe_classifier_dropout=None,


        # step connector 
        sc_num_hidden_layers=2,
        sc_num_attention_heads=12,
        sc_intermediate_size=3072,
        sc_hidden_act="gelu",
        sc_hidden_dropout_prob=0.1,
        sc_attention_probs_dropout_prob=0.1,
        sc_max_position_embeddings=512,
        # sc_initializer_range=0.02,
        sc_layer_norm_eps=1e-12,
        sc_position_embedding_type="absolute",
        # sc_use_cache=True,
        sc_classifier_dropout=None,
        
        # temporal fusor 
        tf_n_layer=4,
        tf_n_head=12,
        tf_n_inner=None,
        tf_activation_function="gelu_new",
        tf_resid_pdrop=0.1,
        tf_embd_pdrop=0.1,
        tf_attn_pdrop=0.1,
        tf_layer_norm_epsilon=1e-5,
        # tf_initializer_range=0.02,
        tf_summary_type="cls_index",
        tf_summary_use_proj=True,
        tf_summary_activation=None,
        tf_summary_proj_to_labels=True,
        tf_summary_first_dropout=0.1,
        tf_scale_attn_weights=True,
        # tf_use_cache=True,
        tf_scale_attn_by_inverse_layer_idx=False,
        tf_reorder_and_upcast_attn=False,
        
        # entry_args = None, # Add this line
        CF_to_CFvocab = None, # Add this line
        OneEntryArgs = None, # Add this line
        **kwargs,
    ):
        
        self.early_fusor_strategy = early_fusor_strategy
        self.late_fusor_strategy = late_fusor_strategy

        self.n_embd = n_embd
        self.initializer_range = initializer_range
        self.use_cache = use_cache

        self.time_field = time_field

        self.lsm_n_embd = n_embd
        self.lsm_n_positions = lsm_n_positions
        self.lsm_n_layer = lsm_n_layer
        self.lsm_n_head = lsm_n_head
        self.lsm_n_inner = lsm_n_inner
        self.lsm_activation_function = lsm_activation_function
        self.lsm_resid_pdrop = lsm_resid_pdrop
        self.lsm_embd_pdrop = lsm_embd_pdrop
        self.lsm_attn_pdrop = lsm_attn_pdrop
        self.lsm_layer_norm_epsilon = lsm_layer_norm_epsilon
        self.lsm_initializer_range = initializer_range
        self.lsm_summary_type = lsm_summary_type
        self.lsm_summary_use_proj = lsm_summary_use_proj
        self.lsm_summary_activation = lsm_summary_activation
        self.lsm_summary_proj_to_labels = lsm_summary_proj_to_labels
        self.lsm_summary_first_dropout = lsm_summary_first_dropout
        self.lsm_scale_attn_weights = lsm_scale_attn_weights
        self.lsm_use_cache = use_cache
        self.lsm_scale_attn_by_inverse_layer_idx = lsm_scale_attn_by_inverse_layer_idx
        self.lsm_reorder_and_upcast_attn = lsm_reorder_and_upcast_attn


        self.fe_hidden_size = n_embd
        self.fe_timestep_lookback = fe_timestep_lookback
        self.fe_timestep_lookahead = fe_timestep_lookahead
        self.fe_embd_pdrop = fe_embd_pdrop
        self.fe_use_field_type_embedding = fe_use_field_type_embedding
        self.fe_num_hidden_layers = fe_num_hidden_layers
        self.fe_num_attention_heads = fe_num_attention_heads
        self.fe_intermediate_size = fe_intermediate_size
        self.fe_hidden_act = fe_hidden_act
        self.fe_hidden_dropout_prob = fe_hidden_dropout_prob
        self.fe_attention_probs_dropout_prob = fe_attention_probs_dropout_prob
        self.fe_max_position_embeddings = fe_max_position_embeddings
        self.fe_initializer_range = initializer_range
        self.fe_layer_norm_eps = fe_layer_norm_eps
        self.fe_position_embedding_type = fe_position_embedding_type
        self.fe_use_cache = use_cache
        self.fe_classifier_dropout = fe_classifier_dropout


        self.sc_hidden_size = n_embd
        self.sc_num_hidden_layers = sc_num_hidden_layers
        self.sc_num_attention_heads = sc_num_attention_heads
        self.sc_intermediate_size = sc_intermediate_size
        self.sc_hidden_act = sc_hidden_act
        self.sc_hidden_dropout_prob = sc_hidden_dropout_prob
        self.sc_attention_probs_dropout_prob = sc_attention_probs_dropout_prob
        self.sc_max_position_embeddings = sc_max_position_embeddings
        self.sc_initializer_range = initializer_range
        self.sc_layer_norm_eps = sc_layer_norm_eps
        self.sc_position_embedding_type = sc_position_embedding_type
        self.sc_use_cache = use_cache
        self.sc_classifier_dropout = sc_classifier_dropout


        self.n_layer = tf_n_layer
        self.tf_n_embd = n_embd
        self.tf_n_layer = tf_n_layer
        self.tf_n_head = tf_n_head
        self.tf_n_inner = tf_n_inner
        self.tf_activation_function = tf_activation_function
        self.tf_resid_pdrop = tf_resid_pdrop
        self.tf_embd_pdrop = tf_embd_pdrop
        self.tf_attn_pdrop = tf_attn_pdrop
        self.tf_layer_norm_epsilon = tf_layer_norm_epsilon
        self.tf_initializer_range = initializer_range
        self.tf_summary_type = tf_summary_type
        self.tf_summary_use_proj = tf_summary_use_proj
        self.tf_summary_activation = tf_summary_activation
        self.tf_summary_proj_to_labels = tf_summary_proj_to_labels
        self.tf_summary_first_dropout = tf_summary_first_dropout
        self.tf_scale_attn_weights = tf_scale_attn_weights
        self.tf_use_cache = use_cache
        self.tf_scale_attn_by_inverse_layer_idx = tf_scale_attn_by_inverse_layer_idx
        self.tf_reorder_and_upcast_attn = tf_reorder_and_upcast_attn

        self.OneEntryArgs = OneEntryArgs
        self.CF_to_CFvocab = CF_to_CFvocab
        if OneEntryArgs is not None and CF_to_CFvocab is not None:
            self.initalize_field_info()

        super().__init__(**kwargs)


    def initalize_field_info(self):
        if not hasattr(self, 'OneEntryArgs') or not hasattr(self, 'CF_to_CFvocab'):
            return None 
        
        # self.CF_to_CFvocab = CF_to_CFvocab
        # self.Field_to_CFvocab = None

        # self.set_field_info_with_OneEntryArgs(OneEntryArgs, CF_to_CFvocab)  
        # self.entry_args = entry_args # Add this line

        # self.OneEntryArgs = OneEntryArgs
        # self.CF_to_CFvocab = CF_to_CFvocab
        # print('in set_field_info_with_OneEntryArgs')
        # print(OneEntryArgs)
        # print(CF_to_CFvocab)

        OneEntryArgs = self.OneEntryArgs
        CF_to_CFvocab = self.CF_to_CFvocab
        InputPart = OneEntryArgs['Input_Part']
        TargetField = InputPart['TargetField']
        TimeField = InputPart.get('TimeField', None)
        EventFields = InputPart.get('EventFields', [])


        CF_list = InputPart['CF_list']  

        if TimeField is not None:
            FieldList = [TimeField] + EventFields
        else:
            FieldList = EventFields
        # FieldList

        Field_to_CFs = {Field: [CF for CF in CF_list if Field in CF] for Field in FieldList}
        # Field_to_CFs

        Field_to_CFvocab = {Field: CF_to_CFvocab[CFs[0]] for Field, CFs in Field_to_CFs.items()}
        # Field_to_CFvocab

        # self.Field_to_CFvocab = Field_to_CFvocab  
        field_to_fieldinfo = {}

        for field in FieldList:
            tkn2tid = Field_to_CFvocab[field]['input_ids']['tkn2tid']
            # field_to_vocabsize = {field: len(tkn2tid)}
            vocab_size = len(tkn2tid) 
            bos_token_id = tkn2tid['[BOS]']
            eos_token_id = tkn2tid['[EOS]']
            pad_token_id = 0
            field_to_fieldinfo[field] = {
                'vocab_size': vocab_size,
                'bos_token_id': bos_token_id,
                'eos_token_id': eos_token_id,
                'pad_token_id': pad_token_id,
            }

        self.field_to_fieldinfo = field_to_fieldinfo


        TargetField_CFs = [CF for CF in CF_list if TargetField in CF] 
        target_field_vocab = CF_to_CFvocab[TargetField_CFs[0]]
        self.target_field_vocab = target_field_vocab
        tkn2tid = target_field_vocab['input_ids']['tkn2tid']   
        self.lsm_vocab_size = len(tkn2tid)
        self.lsm_bos_token_id = tkn2tid['[BOS]']
        self.lsm_eos_token_id = tkn2tid['[EOS]']
        self.lsm_pad_token_id = 0

        
    def to_dict(self):
        output = super().to_dict()
        
        # List of fields to exclude
        fields_to_exclude = ['CF_to_CFvocab', 'target_field_vocab', 'OneEntryArgs']
        
        # Remove excluded fields if they exist
        for field in fields_to_exclude:
            if field in output:
                del output[field]
                
        return output



class CgmLhmOnnxConfig(OnnxConfigWithPast):
    def __init__(
        self,
        config: PretrainedConfig,
        task: str = "default",
        patching_specs: List[PatchingSpec] = None,
        use_past: bool = False,
    ):
        super().__init__(config, task=task, patching_specs=patching_specs, use_past=use_past)
        if not getattr(self._config, "pad_token_id", None):
            # TODO: how to do that better?
            self._config.pad_token_id = 0

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        common_inputs = OrderedDict({"input_ids": {0: "batch", 1: "sequence"}})
        if self.use_past:
            self.fill_with_past_key_values_(common_inputs, direction="inputs")
            common_inputs["attention_mask"] = {0: "batch", 1: "past_sequence + sequence"}
        else:
            common_inputs["attention_mask"] = {0: "batch", 1: "sequence"}

        return common_inputs

    @property
    def num_layers(self) -> int:
        return self._config.n_layer

    @property
    def num_attention_heads(self) -> int:
        return self._config.n_head

    def generate_dummy_inputs(
        self,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
    ) -> Mapping[str, Any]:
        common_inputs = super(OnnxConfigWithPast, self).generate_dummy_inputs(
            tokenizer, batch_size=batch_size, seq_length=seq_length, is_pair=is_pair, framework=framework
        )

        # We need to order the input in the way they appears in the forward()
        ordered_inputs = OrderedDict({"input_ids": common_inputs["input_ids"]})

        # Need to add the past_keys
        if self.use_past:
            if not is_torch_available():
                raise ValueError("Cannot generate dummy past_keys inputs without PyTorch installed.")
            else:
                import torch

                batch, seqlen = common_inputs["input_ids"].shape
                # Not using the same length for past_key_values
                past_key_values_length = seqlen + 2
                past_shape = (
                    batch,
                    self.num_attention_heads,
                    past_key_values_length,
                    self._config.hidden_size // self.num_attention_heads,
                )
                ordered_inputs["past_key_values"] = [
                    (torch.zeros(past_shape), torch.zeros(past_shape)) for _ in range(self.num_layers)
                ]

        ordered_inputs["attention_mask"] = common_inputs["attention_mask"]
        if self.use_past:
            mask_dtype = ordered_inputs["attention_mask"].dtype
            ordered_inputs["attention_mask"] = torch.cat(
                [ordered_inputs["attention_mask"], torch.ones(batch, past_key_values_length, dtype=mask_dtype)], dim=1
            )

        return ordered_inputs

    @property
    def default_onnx_opset(self) -> int:
        return 13