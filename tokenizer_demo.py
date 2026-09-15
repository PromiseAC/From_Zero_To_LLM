from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-0.5B"
)

texts = [
    "我喜欢机器学习",
    "I really like machine learning"
]

for text in texts:
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.encode(text)
    decoded = tokenizer.decode(token_ids)

    print("text:", text)
    print("tokens:", tokens)
    print("token ids:", token_ids)
    print("vocab size:", tokenizer.vocab_size)
    print("decoded:", decoded)

print("special tokens map:")
print(tokenizer.special_tokens_map)

print()

print("bos token:", tokenizer.bos_token)
print("bos token id:", tokenizer.bos_token_id)

print("eos token:", tokenizer.eos_token)
print("eos token id:", tokenizer.eos_token_id)

print("pad token:", tokenizer.pad_token)
print("pad token id:", tokenizer.pad_token_id)

print("unk token:", tokenizer.unk_token)
print("unk token id:", tokenizer.unk_token_id)

print()

print("all special tokens:")
print(tokenizer.all_special_tokens)

print("all special ids:")
print(tokenizer.all_special_ids)

print("tokenizer.vocab_size:", tokenizer.vocab_size)
print("len(tokenizer):", len(tokenizer))