import torch
from torch import nn
from transformers import GPT2Config, GPT2Model, GPT2PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import GPT2LMHeadModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions



class GPT2WithHMForSequenceClassification(GPT2PreTrainedModel):
    def __init__(self, config: GPT2Config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.transformer = GPT2Model(config)
        self.hm_embeddings = nn.Embedding(config.hm_vocab_size, config.hidden_size)
        self.hm_dropout = nn.Dropout(config.embd_pdrop)
        self.score = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        hm_ids=None,
        labels=None,
        **kwargs
    ):
        # 1) Word and positional embeddings
        inputs_embeds = self.transformer.wte(input_ids)
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.size(-1), dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        position_embeds = self.transformer.wpe(position_ids)
        inputs_embeds = inputs_embeds + position_embeds

        # 2) Add HM embeddings
        hm_embeds = self.hm_embeddings(hm_ids)
        hm_embeds = self.hm_dropout(hm_embeds)
        inputs_embeds = inputs_embeds + hm_embeds

        # 3) Run through GPT2
        transformer_outputs = self.transformer(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True
        )
        hidden_states = transformer_outputs.last_hidden_state

        # 4) Classification head on last token ([EOS] token logic)
        logits = self.score(hidden_states[:, -1, :])

        # 5) Loss
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


class GPT2WithHMForCausalLM(GPT2LMHeadModel):
    def __init__(self, config: GPT2Config):
        super().__init__(config)
        self.hm_embeddings = nn.Embedding(config.hm_vocab_size, config.hidden_size)
        self.hm_dropout = nn.Dropout(config.embd_pdrop)
        self._init_weights(self.hm_embeddings)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        hm_ids=None,
        labels=None,
        **kwargs
    ):
        # 1) Embed tokens and positions
        inputs_embeds = self.transformer.wte(input_ids)
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.size(-1), dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        position_embeds = self.transformer.wpe(position_ids)
        inputs_embeds = inputs_embeds + position_embeds

        # 2) Add HM embeddings
        hm_embeds = self.hm_embeddings(hm_ids)
        hm_embeds = self.hm_dropout(hm_embeds)
        inputs_embeds = inputs_embeds + hm_embeds

        # 3) Normalize after fusion
        # inputs_embeds = self.drop(self.ln_f(inputs_embeds))
        # inputs_embeds = self.transformer.ln_f(inputs_embeds)  # LayerNorm (same as GPT-2 post-embedding norm)
        inputs_embeds = self.transformer.drop(self.transformer.ln_f(inputs_embeds))


        # 3) Forward to transformer
        transformer_outputs = self.transformer(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )

        hidden_states = transformer_outputs.last_hidden_state
        lm_logits = self.lm_head(hidden_states)

        # 4) Loss computation
        loss = None
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        # Support past-key values and step-wise generation
        

        # TODO: in the future, support use_cache
        
        input_shape = input_ids.shape

        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        position_ids = kwargs.get("position_ids", None)
        if position_ids is None:
            # GPT2 uses sequential positions
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "hm_ids": kwargs.get("hm_ids", None),  # Pass through custom field
            "use_cache": kwargs.get("use_cache", True),
        }
