from transformers import AutoTokenizer
from pathlib import Path

model_dir = (
    Path(__file__).resolve().parent.parent
    / "minimind"
    / "model"
)

tokenizer = AutoTokenizer.from_pretrained(
    model_dir
)

texts = [
    "中国的首都是北京。",
    "机器学习是人工智能的重要分支。",
    "Hello MiniMind."
]

for text in texts:

    tokens = tokenizer.tokenize(text)

    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False
    )

    decoded_text = tokenizer.decode(
        token_ids,
        skip_special_tokens=False
    )


    print("=" * 50)
    print("raw text:", text)
    print("tokens:", tokens)
    print("token ids:", token_ids)
    print("decoded text:", decoded_text)

    print("token pieces:")

    for token_id in token_ids:
        piece = tokenizer.decode(
            [token_id],
            skip_special_tokens=False
        )

        print(
            token_id,
            "->",
            repr(piece)
        )

print("=" * 50)
print("bos token:", tokenizer.bos_token)
print("bos token id:", tokenizer.bos_token_id)

print("eos token:", tokenizer.eos_token)
print("eos token id:", tokenizer.eos_token_id)

print("pad token:", tokenizer.pad_token)
print("pad token id:", tokenizer.pad_token_id)

print("unk token:", tokenizer.unk_token)
print("unk token id:", tokenizer.unk_token_id)

print("special tokens map:")
print(tokenizer.special_tokens_map)

text = "中国的首都是北京。"

ids_without_special = tokenizer.encode(
    text,
    add_special_tokens=False
)

ids_with_special = tokenizer.encode(
    text,
    add_special_tokens=True
)

print("=" * 50)

print(
    "without special:",
    ids_without_special
)

print(
    "with special:",
    ids_with_special
)

print(
    "without decoded:",
    tokenizer.decode(
        ids_without_special,
        skip_special_tokens=False
    )
)

print(
    "with decoded:",
    tokenizer.decode(
        ids_with_special,
        skip_special_tokens=False
    )
)