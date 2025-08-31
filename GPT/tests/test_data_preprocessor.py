import pytest
import torch


# ==== Unit Tests ====
def test_data_tensor_created(preprocessor, model):

    # Create the data tensor
    preprocessor.create_data_tensor(model)
    assert preprocessor.data_tensor is not None and preprocessor.tensor_size > 0, "The data_tensor was not created."

def test_dataset_splits_generated(preprocessor, model):
    
    # Creating the data tensor
    preprocessor.create_data_tensor(model)

    # Perform the splits on the set
    train, valid, test = preprocessor.train_valid_test()

    # Checking the existence of splits
    assert train is not None, "Train set was not generated."
    assert valid is not None, "Valid set was not generated."
    assert test is not None, "Test set was not generated."

    # Checking the size of splits
    assert len(train) - 2 == int(preprocessor.tensor_size * 0.85), "Train set size is wrong."
    assert len(valid) == int(preprocessor.tensor_size * 0.10), "Valid set size is wrong."
    assert len(test) == int(preprocessor.tensor_size * 0.05), "Test set size is wrong."

@pytest.mark.parametrize(
    "batch_size, block_size, shape_expected",
    [
        (16, 8, (16, 8)),
        (32, 16, (32, 16)),
        (64, 128, (64, 128)),
        (128, 256, (128, 256))
    ]
)
def test_get_batch_works(preprocessor, model, batch_size, block_size, shape_expected):

    # Creating the data tensor
    preprocessor.create_data_tensor(model)
    
    # Acquiring the splits
    train, valid, test = preprocessor.train_valid_test()
    
    # Testing the batches
    batch_X, batch_y = preprocessor.get_batch(train, batch_size, block_size)
    out_X, out_y = torch.tensor(batch_X.shape), torch.tensor(batch_y.shape)

    assert torch.allclose(out_X, torch.tensor([shape_expected])), "Wrong batch shape was generated for features."
    assert torch.allclose(out_y, torch.tensor([shape_expected])), "Wrong batch shape was generated for targets."
