from src.data import IMDBMovieReview
from src.data_preprocess import DataPreprocessor
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

@pytest.fixture
def preprocessor(dataset):
    """Fixture to load the datapreprocessor instance."""
    data_string = dataset.refine_structure()
    return DataPreprocessor(data_string)
