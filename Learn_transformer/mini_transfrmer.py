import torch
import torch.nn as nn

from transformer_components import RMSNorm
from transformer_block import TransformerBlock


class MiniTransformer(nn.Module):
    def __init__(
            self,
            vocab_size,
            d_model,
            num_heads,
            num_layers,
            intermediate_size,
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    intermediate_size=intermediate_size,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm = RMSNorm(d_model)

        self.lm_head = nn.Linear(d_model, vocab_size)


    def forward(self, tokens):
        x = self.token_embedding(tokens)

        for layer in self.layers:
            x, weights = layer(x)

        x = self.norm(x)
        logits = self.lm_head(x)

        return logits


if __name__ == "__main__":
    batch_size = 2
    sequence_length = 8
    vocab_size = 100
    d_model = 64
    num_heads = 4
    num_layers = 2
    intermediate_size = 256

    tokens = torch.randint(
        0,
        vocab_size,
        (batch_size, sequence_length),
    )

    model = MiniTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        intermediate_size=intermediate_size,
    )

    logits = model(tokens)

    print("tokens shape:", tokens.shape)
    print("logits shape:", logits.shape)
