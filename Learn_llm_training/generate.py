import torch
import sys
from pathlib import Path

transformer_dir = (
    Path(__file__).resolve().parent.parent
    / "Learn_transformer"
)

sys.path.insert(0, str(transformer_dir))

from mini_transformer import MiniTransformer

def generate(model, idx, max_new_tokens):
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(idx)

        next_token_logits = logits[:, -1, :]
        next_token_id = torch.argmax(
            next_token_logits,
            dim=-1,
            keepdim=True
        )

        idx = torch.cat(
            (idx, next_token_id),
            dim=1
        )
    return idx

prompt = "语言模型"

checkpoint_dir = Path(__file__).resolve().parent
checkpoint_paths = [
    checkpoint_dir / "checkpoint_epoch_0.pt",
    checkpoint_dir / "checkpoint_epoch_3.pt",
    checkpoint_dir / "checkpoint_epoch_9.pt",
]

for checkpoint_path in checkpoint_paths:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu"
    )

    print(
        checkpoint_path.name,
        "epoch =", checkpoint["epoch"],
        "global_step =", checkpoint["global_step"]
    )


    stoi = checkpoint["stoi"]
    itos = checkpoint["itos"]

    model = MiniTransformer(
        vocab_size=checkpoint["vocab_size"],
        d_model=checkpoint["d_model"],
        num_heads=checkpoint["num_heads"],
        num_layers=checkpoint["num_layers"],
        intermediate_size=checkpoint["intermediate_size"]
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    unk_id = stoi["<UNK>"]

    prompt_ids = [
        stoi.get(ch, unk_id)
        for ch in prompt
    ]

    idx = torch.tensor(
        prompt_ids,
        dtype=torch.long
    )

    idx = idx.unsqueeze(0)

    generated_ids = generate(
        model,
        idx,
        max_new_tokens=10
    )

    token_ids = generated_ids[0].tolist()

    generated_text = "".join(
        itos[token_id]
        for token_id in token_ids
    )

    print(
            f"epoch={checkpoint['epoch']} "
            f"global_step={checkpoint['global_step']} "
            f"generated={generated_text}"
        )