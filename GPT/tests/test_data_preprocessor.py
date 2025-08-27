# ==== Unit Tests ====
def test_splits_generated(preprocessor, model):
    """Checks for the existance of the split datasets."""

    # Create the data tensor
    preprocessor.create_data_tensor(model)
    
    # Perform the splits on the set
    train, valid, test = preprocessor.train_valid_test()

    assert train is not None and len(train) > 0, "Train set was not generated."
    assert valid is not None and len(valid) > 0, "Valid set was not generated."
    assert test is not None and len(test) > 0, "Test set was not generated."

