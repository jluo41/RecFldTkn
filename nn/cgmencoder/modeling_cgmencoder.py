import torch
from torch import nn
from transformers import RobertaConfig, RobertaForSequenceClassification
from transformers.models.roberta.modeling_roberta import SequenceClassifierOutput
from transformers import RobertaConfig, RobertaForMaskedLM
from transformers.models.roberta.modeling_roberta import MaskedLMOutput


class RobertaWithHMForSequenceClassification(RobertaForSequenceClassification):
    def __init__(self, config: RobertaConfig):
        super().__init__(config)
        self.hm_embeddings = nn.Embedding(config.hm_vocab_size, config.hidden_size)
        self.hm_dropout = nn.Dropout(config.hidden_dropout_prob)
        self._init_weights(self.hm_embeddings)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        hm_ids=None,
        labels=None,
        **kwargs
    ):
        # 1) Build token+pos+type embeddings
        inputs_embeds = self.roberta.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )

        # 2) Add hm embeddings
        hm_embeds = self.hm_embeddings(hm_ids)
        hm_embeds = self.hm_dropout(hm_embeds)
        inputs_embeds = inputs_embeds + hm_embeds

        # Apply LayerNorm and Dropout as in the original Roberta
        inputs_embeds = self.roberta.embeddings.LayerNorm(inputs_embeds)
        inputs_embeds = self.roberta.embeddings.dropout(inputs_embeds)

        # 3) Run the full RobertaModel
        model_outputs  = self.roberta(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )
        sequence_output = model_outputs.last_hidden_state  # [batch, seq_len, hidden]

        # 4) Classifier wants the sequence, not the pooled vector
        logits = self.classifier(sequence_output)          # picks out token 0 internally

        # 5) Loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss     = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=model_outputs.hidden_states,
            attentions=model_outputs.attentions,
        )



class RobertaWithHMForMaskedLM(RobertaForMaskedLM):
    def __init__(self, config: RobertaConfig):
        super().__init__(config)
        self.hm_embeddings = nn.Embedding(config.hm_vocab_size, config.hidden_size)
        self.hm_dropout = nn.Dropout(config.hidden_dropout_prob)
        self._init_weights(self.hm_embeddings)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        hm_ids=None,
        labels=None,
        **kwargs
    ):
        # 1) Build token+pos+type embeddings
        inputs_embeds = self.roberta.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )

        # 2) Add hm embeddings
        hm_embeds = self.hm_embeddings(hm_ids)
        hm_embeds = self.hm_dropout(hm_embeds)
        inputs_embeds = inputs_embeds + hm_embeds

        # Using a fusion layer (e.g., MLP or gated mechanism) instead of raw addition:
        # fused_embeds = self.fusion_layer(torch.cat([inputs_embeds, hm_embeds], dim=-1))

        # Apply LayerNorm and Dropout as in the original Roberta
        inputs_embeds = self.roberta.embeddings.LayerNorm(inputs_embeds)
        inputs_embeds = self.roberta.embeddings.dropout(inputs_embeds)

        # 3) Run through full Roberta model
        outputs = self.roberta(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )
        sequence_output = outputs.last_hidden_state

        # 4) MLM head
        prediction_scores = self.lm_head(sequence_output)

        # 5) Compute loss if labels are provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Flatten the tokens
            loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        return MaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
