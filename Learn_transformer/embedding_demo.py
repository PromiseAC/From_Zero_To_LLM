import torch
import torch.nn as nn

BATCH_SIZE = 2
SEQ_LEN = 4
VOCAB_SIZE = 10
HIDDEN_SIZE = 8

def main():
    #[B, T]
    token_ids = torch.tensor([
        [1, 5, 8, 3],
        [7, 2, 9, 4],
    ])
    print("token_ids:")
    print(token_ids)
    print("token_ids shape:", token_ids.shape)

    #Embedding层
    embedding = nn.Embedding(
        num_embeddings=VOCAB_SIZE,
        embedding_dim=HIDDEN_SIZE,
    )
    #[B,T]->[B,T,D]
    x = embedding(token_ids)

    print("\nembedding output shape:")
    print(x.shape)

    #模拟 LM Head
    lm_head = nn.Linear(
        HIDDEN_SIZE,
        VOCAB_SIZE,
    )

    # [B,T,D] -> [B,T,V]
    logits = lm_head(x)

    print("\nlogits shape:")
    print(logits.shape)

if __name__ == "__main__":
    main()