import matplotlib.pyplot as plt

from src.data import DataHandler
from src.resnet50 import Resnet50


def main():
    """The main function for the Vision Transformer."""
    print("Hello, Vision Transformer")

    data_handle = DataHandler()

    # Loading the sets from disk
    train_set = data_handle.load_set("train")
    valid_set = data_handle.load_set("valid")

    # Preparing the sets
    train_prep = data_handle.prepare_dataset(train_set)
    valid_prep = data_handle.prepare_dataset(valid_set)

    # Loading the Model and viewing the Architecture
    first_resnet_50 = Resnet50()
    print(first_resnet_50)


# ==== Driver Code ====
if __name__ == "__main__":
    main()