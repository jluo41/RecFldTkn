from transformers import GPT2Config

class GPT2WithHMConfig(GPT2Config):
    model_type = "gpt2"  # keep HF happy

    def __init__(self, hm_vocab_size=288, **kwargs):
        super().__init__(**kwargs)
        self.hm_vocab_size = hm_vocab_size
