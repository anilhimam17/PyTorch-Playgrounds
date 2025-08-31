import torch

from src.multihead_attention import MultiHeadAttention


def main():
    sample_X = torch.randn(size=(128, 32, 64))

    n_embd = 64
    num_heads = 4
    block_size = 32

    mha_block = MultiHeadAttention(
        num_heads=num_heads,
        head_size=n_embd//num_heads,
        block_size=block_size,
        n_embd=n_embd
    )

    attention_out = mha_block(sample_X)
    print(attention_out.shape)
    print(sample_X.shape)
    

main()
