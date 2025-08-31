# ==== Unit Tests ====
def test_dataset_load_works(dataset):
    assert dataset is not None, "The dataset was not loaded."
    assert "sentiment" not in dataset.dataframe.columns, "The sentiment column was not removed."

def test_datastring_generated(dataset):
    assert dataset.dataset_string is not None, "The datastring was not generated."

def test_vocabulary_created(dataset):
    assert dataset.vocab is not None and len(dataset.vocab) == 77, "The vocabulary was not generated or vocab size does not match."
