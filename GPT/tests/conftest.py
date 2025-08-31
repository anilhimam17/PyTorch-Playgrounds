import pytest

from src.data import IMDBMovieReview
from src.data_preprocess import DataPreprocessor
from src.self_attention import SelfAttentionHead
from src.multihead_attention import MultiHeadAttention
from src.my_gpt import MyGPT


# ==== Common PyTest Fixtures for all the tests ====
@pytest.fixture
def dataset():
    """Fixture to download and load the dataset."""
    imdb = IMDBMovieReview()
    imdb.refine_structure()
    return imdb

@pytest.fixture
def model(dataset):
    """Fixture to load the model with the vocab."""
    return MyGPT(vocab=dataset.vocab)

@pytest.fixture
def preprocessor(dataset):
    """Fixture to load the datapreprocessor instance."""
    data_string = dataset.refine_structure()
    return DataPreprocessor(data_string)

@pytest.fixture
def self_attention_head():
    """Fixture to load a new self-attention block instance."""
    attention_head = SelfAttentionHead(head_size=256, input_features=64, block_size=32)
    return attention_head

@pytest.fixture
def multihead_attention_block():
    """Fixture to load a new Multihead Attention block instance."""

    # Multihead Attention Block Parameters
    n_embd = 64
    num_heads = 4
    block_size = 32

    mha_block = MultiHeadAttention(
        num_heads=num_heads,
        head_size=n_embd//num_heads,
        block_size=block_size,
        n_embd=n_embd
    )

    return mha_block
