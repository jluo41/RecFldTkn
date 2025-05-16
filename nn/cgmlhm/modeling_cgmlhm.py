import math
import inspect
import copy 
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Tuple, Union
from torch.cuda.amp import autocast
import torch
import torch.utils.checkpoint
from torch import nn
from torch.cuda.amp import autocast
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel, SequenceSummary
from transformers.pytorch_utils import Conv1D, find_pruneable_heads_and_indices, prune_conv1d_layer
from transformers.utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling

from transformers import PreTrainedModel, GenerationMixin



from .modeling_cgmlsm import AttnBlock, CgmLsmModel
from .modeling_fieldencoder import TfmEncoder, Pooler, FieldEncoderModel
from .configuration_cgmlhm import CgmLhmConfig
from .configuration_cgmlsm import CgmLsmConfig
from .configuration_fieldencoder import FieldEncoderConfig

logger = logging.get_logger(__name__)


class StepConnector(nn.Module):

    def __init__(self, step_config):
        super().__init__()
        self.num_hidden_layers = step_config.num_hidden_layers
        if step_config.num_hidden_layers > 0:
            self.encoder = TfmEncoder(step_config)

        # Pooler
        self.pooler = Pooler(step_config)

    def forward(
        self,
        step_field_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # input_vectors shape: (batch_size, length_of_fields, field_vector)
        batch_size, seq_len, field_num, hidden_dim = step_field_states.shape
        hidden_states = step_field_states.contiguous().view(-1, field_num, hidden_dim)

        # use encoder or not. 
        if self.num_hidden_layers:
            encoder_output = self.encoder(hidden_states)
            last_hidden_state = encoder_output.last_hidden_state   
        else:
            last_hidden_state = hidden_states
        
    
        # Reshape back to original
        last_hidden_state = last_hidden_state.view(batch_size, seq_len, field_num, hidden_dim)
        pooled_output = self.pooler(last_hidden_state)

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output
        )
                                            

class CgmLhmPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = CgmLhmConfig
    # model_type = CgmLhmConfig.model_type
    # load_tf_weights = load_tf_weights_in_gpt2
    base_model_prefix = "lhm" # previous it is transformer
    is_parallelizable = True
    supports_gradient_checkpointing = True
    _no_split_modules = ["AttnBlock"]
    _skip_keys_device_placement = "past_key_values"

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name == "c_proj.weight":
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                p.data.normal_(mean=0.0, std=(self.config.initializer_range / math.sqrt(2 * self.config.n_layer)))


class CgmLhmModel(CgmLhmPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)


        # -------- set up the lsm --------
        lsm_kwargs = {k.split('lsm_')[1]: v for k, v in config.to_dict().items() if 'lsm_' in k}
        lsm_config = CgmLsmConfig(**lsm_kwargs)
        self.lsm_config = lsm_config    
        self.lsm_model = CgmLsmModel(lsm_config)  

        # -------- set up the field encoders --------
        fe_kwargs = {k.split('fe_')[1]: v for k, v in config.to_dict().items() if 'fe_' in k}
        
        if len(config.field_to_fieldinfo) > 0:
            self.field_to_feconfig = {}
            self.field_encoders = torch.nn.ModuleDict()
            for field, fieldinfo in config.field_to_fieldinfo.items():
                fe_config = FieldEncoderConfig(**fieldinfo, **fe_kwargs) 
                field_encoder = FieldEncoderModel(fe_config) # this could be pretrained. 
                self.field_to_feconfig[field] = fe_config
                self.field_encoders[field] = field_encoder
        else:
            self.field_to_feconfig = {}
            self.field_encoders = None

        self.time_field = config.time_field 

        # -------- set up the step connector --------
        sc_kwargs = {k.split('sc_')[1]: v for k, v in config.to_dict().items() if 'sc_' in k}
        sc_config = FieldEncoderConfig(**sc_kwargs)
        self.sc_config = sc_config
        if len(self.field_to_feconfig) > 0:
            self.step_connector_model = StepConnector(sc_config)
        else:
            self.step_connector_model = None

        # -------- set up the temporal fusor --------
        tf_kwargs = {k.split('tf_')[1]: v for k, v in config.to_dict().items() if 'tf_' in k}
        tf_config = CgmLsmConfig(**tf_kwargs)
        self.tf_config = tf_config
        if tf_config.num_hidden_layers > 0:
            self.temporal_fusor_model = nn.ModuleList([AttnBlock(tf_config, layer_idx=i) for i in range(tf_config.num_hidden_layers)])
            self.embed_dim = config.n_embd 
            self.ln_f = nn.LayerNorm(self.embed_dim, eps=tf_config.layer_norm_epsilon)
        else:
            self.temporal_fusor_model = None
            self.ln_f = None

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()


    def get_input_embeddings(self):
        return self.lsm_model.wte

    def set_input_embeddings(self, new_embeddings):
        self.lsm_model.wte = new_embeddings
        

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        # past_key_values_lsm: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[Tuple[torch.Tensor]]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        # token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        # total_field_info: Optional[dict] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        
        
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        ############################ output settings ############################
        # step 1: things to return
        # output_attentions: True or False
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        # output_hidden_states: True or False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        
        # use_cache: True of False
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        
        # return_dict: True or False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        
        ############################ input information ############################
        # step 2: input_ids (or input_embeds)
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            raise ValueError("Current CgmLhm doesn't support inputs_embeds")
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        
        # step 6: in the future, take care of the past_key_values.

        ##############
        if past_key_values is not None:
            past_key_values_lsm = past_key_values[0]
            past_key_values = past_key_values[1]
        else:
            past_key_values_lsm = None
            past_key_values = None
        ##############


        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * self.tf_config.num_hidden_layers)
        else:
            past_length = past_key_values[0][0].size(-2)
        # past_length: how many steps in the past we are keeping
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)
        
        
        ############################ get the results ############################
        step_state_list = []
        field_to_feconfig = self.field_to_feconfig
        # step_state_list = [target_state]
        field_to_encoder_outputs = {}

        for field, fe_config in field_to_feconfig.items():
            
            # if the field is the time field, skip it. 
            # if field == time_field: continue

            batch_field_inputs = {k.split('--')[1]: v for k, v in kwargs.items() if field + '--' in k}

            ############################################################
            # if the field is not TimeField, and there is no event indicators, skip this field.
            if 'event_indicators' in batch_field_inputs:
                event_indicators = batch_field_inputs['event_indicators'] 
                exists_event_indicators = bool(event_indicators.sum() > 0)
                if not exists_event_indicators: continue 
            else:
                exists_event_indicators = False
                assert field == self.time_field
            ############################################################

            # print('\n===================')
            field_encoder = self.field_encoders[field] 

            # print('reshape')
            for value_name, values in batch_field_inputs.items():
                # print('before', value_name, values.shape)
                a, b = values.size(0), values.size(1)
                # print(a, b)
                values = values.reshape(a * b, -1)
                batch_field_inputs[value_name] = values
                # print(value_name, values.shape)

            # here you have a newly reshaped event_indicators. 
            if exists_event_indicators:
                # this event_indicators is the newly reshaped event_indicators. 
                event_indicators = batch_field_inputs['event_indicators'] 
                mask = event_indicators.bool().squeeze() 
                batch_field_inputs_filtered = {}
                for k, v in batch_field_inputs.items():
                    if k == 'event_indicators': continue
                    batch_field_inputs_filtered[k] = v[mask]
                batch_field_inputs = batch_field_inputs_filtered
                # print([k for k in batch_field_inputs.keys()])

            field_input_ids = batch_field_inputs['input_ids']
            field_attention_mask = field_input_ids.ne(fe_config.pad_token_id)
            batch_field_inputs['attention_mask'] = field_attention_mask

            # ----------------- encoder the field -----------------
            # print('\nfield_encoder')
            field_outputs = field_encoder(**batch_field_inputs)
            hidden_state  = field_outputs.last_hidden_state#.shape
            pooler_output = field_outputs.pooler_output# .shape
            # print('field_outputs.hidden_state', hidden_state.shape)
            # print('field_outputs.pooler_output', pooler_output.shape)

            # hidden_state
            if exists_event_indicators:
                hidden_state_origin = torch.zeros([len(mask),] + list(hidden_state.shape[1:])).to(device)
                
                # logger.info(f'hidden_state_origin.device: {hidden_state_origin.device}')
                # logger.info(f'hidden_state.device: {hidden_state.device}')
                # logger.info(f'mask.device: {mask.device}')

                hidden_state_origin[mask] = hidden_state
                pooler_output_origin = torch.zeros([len(mask),] + list(pooler_output.shape[1:])).to(device)
                pooler_output_origin[mask] = pooler_output
            else:
                hidden_state_origin  = hidden_state
                pooler_output_origin = pooler_output

            new_shape = [a, b] + list(hidden_state.shape[1:])
            hidden_state_origin = hidden_state_origin.reshape(new_shape)
            # print('field_outputs.hidden_state_origin', hidden_state_origin.shape)

            new_shape = [a, b] + list(pooler_output.shape[1:])
            pooler_output_origin = pooler_output_origin.reshape(new_shape)
            # print('field_outputs.pooler_output_origin', pooler_output_origin.shape)

            field_to_encoder_outputs[field] = {
                'hidden_state': hidden_state_origin,
                'pooler_output': pooler_output_origin,
            }
            step_state_list.append(pooler_output_origin) 


        ############################ get the results ############################
        # if len(step_state_list) > 0:
        #     # get step_state, this should be the encoder. 
        #     # the field number in step_field_states could be 2, or 3, or 4, depending on the available fields. 
        #     step_field_states = torch.stack([target_state] + step_state_list, dim=1)
        #     step_output = self.step_connector_model(step_field_states)
        #     step_states = step_output.pooler_output
        # else:
        #     step_states = target_state


        ############################ target_field: get the results ############################
        input_embeds = self.lsm_model.wte(input_ids)

        if len(step_state_list) > 0 and self.config.early_fusor_strategy == True:
            for step_state in step_state_list:
                input_embeds = input_embeds + step_state[:,:input_embeds.size(1),:]

        lsm_model_inputs = {
            # 'input_ids': input_ids,
            'inputs_embeds': input_embeds,
            'past_key_values': past_key_values_lsm,
            # "use_cache": True,
        }


        lsm_outputs = self.lsm_model(**lsm_model_inputs)
        target_state = lsm_outputs.last_hidden_state
        past_key_values_lsm = lsm_outputs.past_key_values
        
        if len(step_state_list) > 0 and self.config.late_fusor_strategy == True:
            # step_states = target_state #  + target_state
            step_field_states = torch.stack([target_state] + step_state_list, dim=1)
            step_output = self.step_connector_model(step_field_states)
            step_states = step_output.pooler_output
        else:
            step_states = target_state


        ############################ enter GPT2 Models and calculate hidden_states ############################
        hidden_states = step_states
        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)
        
        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        
        if self.tf_config.num_hidden_layers > 0:
            ############################ prepare tensors for GPT2 Models ############################
            # GPT2Attention mask.
            if attention_mask is not None:
                if batch_size <= 0:
                    raise ValueError("batch_size has to be defined and > 0")
                attention_mask = attention_mask.view(batch_size, -1)
                attention_mask = attention_mask[:, None, None, :]
                attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
                attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

            if self.config.add_cross_attention and encoder_hidden_states is not None:
                encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
                encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
                if encoder_attention_mask is None:
                    encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
                encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
            else:
                encoder_attention_mask = None

            head_mask = self.get_head_mask(head_mask, self.config.n_layer)
            if self.gradient_checkpointing and self.training:
                if use_cache:
                    logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")
                    use_cache = False

            for i, (block, layer_past) in enumerate(zip(self.temporal_fusor_model, past_key_values)):
                # Model parallel
                if self.model_parallel:
                    torch.cuda.set_device(hidden_states.device)
                    # Ensure layer_past is on same device as hidden_states (might not be correct)
                    if layer_past is not None:
                        layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                    # Ensure that attention_mask is always on the same device as hidden_states
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(hidden_states.device)
                    if isinstance(head_mask, torch.Tensor):
                        head_mask = head_mask.to(hidden_states.device)
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                if self.gradient_checkpointing and self.training:
                    outputs = self._gradient_checkpointing_func(
                        block.__call__,
                        hidden_states,
                        None,
                        attention_mask,
                        head_mask[i],
                        encoder_hidden_states,
                        encoder_attention_mask,
                        use_cache,
                        output_attentions,
                    )
                else:
                    outputs = block(
                        hidden_states,
                        layer_past=layer_past,
                        attention_mask=attention_mask,
                        head_mask=head_mask[i],
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=encoder_attention_mask,
                        use_cache=use_cache,
                        output_attentions=output_attentions,
                    )

                #############
                # you will get an **outputs**.
                #############
                hidden_states = outputs[0]
                
                if use_cache is True:
                    presents = presents + (outputs[1],)

                if output_attentions:
                    all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                    if self.config.add_cross_attention:
                        all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

                # Model Parallel: If it's the last layer for that device, put things on the next device
                if self.model_parallel:
                    for k, v in self.device_map.items():
                        if i == v[-1] and "cuda:" + str(k) != self.last_device:
                            hidden_states = hidden_states.to("cuda:" + str(k + 1))
            hidden_states = self.ln_f(hidden_states)

            # make the past_key_values a combination of lsm and temporal_fusor.
            # print('past_key_values_lsm', past_key_values_lsm)
            past_key_values_lhm = presents 
            past_key_values = past_key_values_lsm, past_key_values_lhm

        else:
            # hidden_states = self.ln_f(hidden_states)
            hidden_states = hidden_states 
            past_key_values = past_key_values_lsm, None

        ############################ render output ############################
        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions] if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

class GgmLhmLMHeadModel(CgmLhmPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.lhm = CgmLhmModel(config)

        lsm_config = self.lhm.lsm_config
        self.lm_head = nn.Linear(lsm_config.n_embd, lsm_config.vocab_size, bias=False)

        # Model parallel
        self.model_parallel = False
        self.device_map = None


        self.field_to_feconfig = self.lhm.field_to_feconfig

        # Initialize weights and apply final processing
        self.post_init()

    # check with ChatGPT to understand the following two functions. 
    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, 
                                      input_ids,
                                      past_key_values=None, 
                                      inputs_embeds=None, 
                                      **kwargs):

        # print(f'\n\n========================================= {datetime.now()}')
        # print('length of input_ids', input_ids.shape)
        # print('input_ids', input_ids)
        # past_key_values_lsm = kwargs.get("past_key_values_lsm", None)
        # in this step, make the field_inputs of the same complete_length to the input_ids.

        current_length = input_ids.shape[1] 
        total_field_info = {k: v for k, v in kwargs.items() if '--' in k}.copy()
        for k, v in total_field_info.items():
            total_field_info[k] = v[:, :current_length]
        
        # Omit tokens covered by past_key_values

        # past_key_values_lsm = kwargs.get("past_key_values_lsm", None) 
        if past_key_values is not None:
            past_key_values_lsm = past_key_values[0]
            past_key_values = past_key_values[1] # could be None
        else:
            past_key_values_lsm = None
            past_key_values = None
        
        # if past_key_values_lsm:
        #     print('past_key_values_lsm', past_key_values_lsm[0][0].shape[2])
        # else:
        #     print('past_key_values_lsm is None')

        # should be the shape as:
        # (n_layers, 2, batch_size, n_heads, seq_len, d_kv)
        # - n_layers: Number of layers in the Transformer model.
        # - 2: There are two elements (key and value) for each layer.
        # - batch_size: The number of sequences being processed simultaneously.
        # - n_heads: Number of attention heads.
        # - seq_len: Length of the sequence (cumulative for past tokens).
        # - d_kv: Dimension of each key/value vector (usually hidden_size / n_heads).

        if past_key_values_lsm:
            past_length = past_key_values_lsm[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            # print('the remove_prefix_length is: ' + str(remove_prefix_length))
            # print('complete length of input_ids:', input_ids.shape)
            
            input_ids = input_ids[:, remove_prefix_length:]
            for k, v in total_field_info.items():
                # make sure only the latest ids for the input_ids are used.
                # and they will share the same lengths with input_ids.
                total_field_info[k] = v[:, remove_prefix_length:]

        if past_key_values:
            # print('past_key_values', past_key_values)
            # print('past_key_values_lsm', past_key_values_lsm)
            # print('past_key_values_lsm', past_key_values_lsm[0][0].shape)
            # print(len(past_key_values_lsm), len(past_key_values_lsm[0]))
            # print('past_key_values', past_key_values[0][0].shape)
            # print(len(past_key_values), len(past_key_values[0]))
            assert past_key_values[0][0].shape[2] == past_key_values_lsm[0][0].shape[2]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
            
        # print('model_inputs', model_inputs)


        # concatenate the past_key_values and past_key_values_lsm.
        if past_key_values_lsm:
            past_key_values = past_key_values_lsm, past_key_values

        new_field_inputs = {
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
        }

        model_inputs.update(new_field_inputs)
        model_inputs.update(total_field_info)

        return model_inputs

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values_lsm: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        total_field_info: Optional[dict] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        lhm_outputs = self.lhm(
            input_ids,
            past_key_values=past_key_values,
            past_key_values_lsm=past_key_values_lsm,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            total_field_info = total_field_info,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        hidden_states = lhm_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            # print(loss) 


        if not return_dict:
            output = (lm_logits,) + lhm_outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        '''
        # First forward pass (no past_key_values)
        outputs = self.lhm(input_ids, use_cache=True)
        logits = outputs.logits  # Predictions for next token, you can process the logits for the next token prediction.
        past_key_values = outputs.past_key_values  # Cache for next steps
        '''

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=lhm_outputs.past_key_values,
            hidden_states=lhm_outputs.hidden_states,
            attentions=lhm_outputs.attentions,
            cross_attentions=lhm_outputs.cross_attentions,
        )

    @staticmethod
    def _reorder_cache(
        past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache 
        if [`~PreTrainedModel.beam_search`] or [`~PreTrainedModel.beam_sample`] is called. 
        This is required to match `past_key_values` with the correct beam_idx at every generation step.

        # not used in this version. we only use the greedy search. #
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past_key_values
        )
        
        
    def _validate_model_kwargs(self, model_kwargs):
        """Validates model kwargs for generation. Generate argument typos will also be caught here."""
        # If a `Cache` instance is passed, checks whether the model is compatible with it
        # if isinstance(model_kwargs.get("past_key_values", None), Cache) and not self._supports_cache_class:
        #     raise ValueError(
        #         f"{self.__class__.__name__} does not support an instance of `Cache` as `past_key_values`. Please "
        #         "check the model documentation for supported cache formats."
        #     )

        # Excludes arguments that are handled before calling any model function
        if self.config.is_encoder_decoder:
            for key in ["decoder_input_ids"]:
                model_kwargs.pop(key, None)

        unused_model_args = []
        model_args = set(inspect.signature(self.prepare_inputs_for_generation).parameters)
        # `kwargs`/`model_kwargs` is often used to handle optional forward pass inputs like `attention_mask`. If
        # `prepare_inputs_for_generation` doesn't accept them, then a stricter check can be made ;)
        if "kwargs" in model_args or "model_kwargs" in model_args:
            model_args |= set(inspect.signature(self.forward).parameters)

        # Encoder-Decoder models may also need Encoder arguments from `model_kwargs`
        if self.config.is_encoder_decoder:
            base_model = getattr(self, self.base_model_prefix, None)

            # allow encoder kwargs
            encoder = getattr(self, "encoder", None)
            # `MusicgenForConditionalGeneration` has `text_encoder` and `audio_encoder`.
            # Also, it has `base_model_prefix = "encoder_decoder"` but there is no `self.encoder_decoder`
            # TODO: A better way to handle this.
            if encoder is None and base_model is not None:
                encoder = getattr(base_model, "encoder", None)

            if encoder is not None:
                encoder_model_args = set(inspect.signature(encoder.forward).parameters)
                model_args |= encoder_model_args

            # allow decoder kwargs
            decoder = getattr(self, "decoder", None)
            if decoder is None and base_model is not None:
                decoder = getattr(base_model, "decoder", None)

            if decoder is not None:
                decoder_model_args = set(inspect.signature(decoder.forward).parameters)
                model_args |= {f"decoder_{x}" for x in decoder_model_args}

            # allow assistant_encoder_outputs to be passed if we're doing assisted generating
            if "assistant_encoder_outputs" in model_kwargs:
                model_args |= {"assistant_encoder_outputs"}

        for key, value in model_kwargs.items():
            if value is not None and key not in model_args:
                unused_model_args.append(key)

        # if unused_model_args:
        #     logger.info(
        #         f"The following `model_kwargs` are not used by the model: {unused_model_args} (note: typos in the"
        #         " generate arguments will also show up in this list)"
        #         )
            
