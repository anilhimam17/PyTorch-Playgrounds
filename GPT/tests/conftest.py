from src.data import IMDBMovieReview
from src.my_gpt import MyGPT
import pytest


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