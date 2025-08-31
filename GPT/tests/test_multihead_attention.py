import torch


# ==== Unit Tests ====
def test_mha_block_created(multihead_attention_block):
    assert multihead_attention_block is not None, "There was an issue creating the multihead attention block."

def test_mha_concat_dims(multihead_attention_block):
    
    # Creating a sample input
    sample_X = torch.randn(size=(128, 32, 64))

    # Multihead Attention Propagation
    mha_out = multihead_attention_block(sample_X)
    mha_shape = torch.tensor(mha_out.shape)

    assert torch.allclose(mha_shape, torch.tensor(sample_X.shape)), "The output dimensions of the MHA block are incorrect."
