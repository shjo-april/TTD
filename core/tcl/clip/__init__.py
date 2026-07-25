from .clip import *

class Tokenizer:
    """Wrapper class for CLIP tokenize function."""
    
    def __init__(self, max_seq_len=77, truncate=True):
        self.max_seq_len = max_seq_len
        self.truncate = truncate
    
    def __call__(self, texts):
        expanded_dim = False
        if isinstance(texts, str):
            texts = [texts]
            expanded_dim = True

        result = tokenize(texts, self.max_seq_len, self.truncate)

        if expanded_dim:
            return result[0]

        return result