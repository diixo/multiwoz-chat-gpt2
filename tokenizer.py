
from transformers import GPT2TokenizerFast


class GPT2Tokenizer:

    def __init__(self):

        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

        special_tokens = {
            "bos_token": "<|startoftext|>",
            "eos_token": "<|endoftext|>",
            "cls_token": "<|cls|>",
            "sep_token": "<|sep|>",
            "pad_token": "<|pad|>",
            "additional_special_tokens": ["<|user|>", "<|sys|>"]
        }

        num_added = self.tokenizer.add_special_tokens(special_tokens)

        print(f"Added {num_added} special tokens.")

        self.bos_token, self.bos_token_id = self.tokenizer.bos_token, self.tokenizer.bos_token_id
        self.eos_token, self.eos_token_id = self.tokenizer.eos_token, self.tokenizer.eos_token_id
        self.pad_token, self.pad_token_id = self.tokenizer.pad_token, self.tokenizer.pad_token_id
        self.cls_token, self.cls_token_id = self.tokenizer.cls_token, self.tokenizer.cls_token_id
        self.sep_token, self.sep_token_id = self.tokenizer.sep_token, self.tokenizer.sep_token_id

        self.vocab_size = len(self.tokenizer)


    def tokenize(self, s):
        return self.tokenizer.tokenize(s)

    def encode(self, s):
        return self.tokenizer.encode(s, add_special_tokens=False)

    def decode(self, tok, skip_special_tokens=False):
        return self.tokenizer.decode(tok, skip_special_tokens=skip_special_tokens)

    def __call__(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)


if __name__ == "__main__":

    tokenizer = GPT2Tokenizer()
    # print(tokenizer.tokenize("Hello, how are you?"))
    # print(tokenizer.encode("Hello, how are you?"))
    # print(tokenizer.decode(tokenizer.encode("Hello, how are you?")))
    print(tokenizer.bos_token, tokenizer.bos_token_id)
    print(tokenizer.eos_token, tokenizer.eos_token_id)
    print(tokenizer.cls_token, tokenizer.cls_token_id)
    print(tokenizer.sep_token, tokenizer.sep_token_id)
    print(tokenizer.pad_token, tokenizer.pad_token_id)
    print(tokenizer.vocab_size)
