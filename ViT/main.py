import torch

from src.multi_head_attention import MultiHeadSelfAttention


# The main function for the Vision Transformer
def main():
    print("Hello, Vision Transformer!!!")

    mhsa_block = MultiHeadSelfAttention()
    X = torch.randn(64, 197, 768)

    # Forward Prop
    x = mhsa_block(X)
    print("Attended Embeddings Shape: ", x.shape)


# ==== Driver Code ====
if __name__ == "__main__":
    main()