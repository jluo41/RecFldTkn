from transformers import RobertaConfig

class RobertaWithHMConfig(RobertaConfig):
    model_type = "roberta"       # keep HF happy

    def __init__(self, hm_vocab_size=288, **kwargs):
        super().__init__(**kwargs)
        self.hm_vocab_size = hm_vocab_size
