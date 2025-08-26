# ==== Unit Tests ====
def test_dataset_load_works(dataset):
    assert dataset is not None, "The dataset was not loaded."

def test_datastring_generated(dataset):
    assert len(dataset.vocab) == 99, "The datastring was not generated."
