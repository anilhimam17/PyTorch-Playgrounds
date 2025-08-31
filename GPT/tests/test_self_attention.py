import torch


# ==== Unit Test ====
def test_attention_head_created(self_attention_head):
    assert self_attention_head is not None, "There was a problem in creating the attention head."

def test_attention_dim_preserved(self_attention_head):

    # Creating a random sample input
    sample_X = torch.randn(size=(128, 32, 64))
    attention_out = torch.tensor(self_attention_head(sample_X).shape[:-1])
    
    # Checking the output dims
    assert torch.allclose(attention_out, torch.tensor(sample_X.shape[:-1])), "The B, T dims are not being preserved."
