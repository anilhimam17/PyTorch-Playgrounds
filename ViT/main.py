from src.data import DataHandler
import matplotlib.pyplot as plt


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


# ==== Driver Code ====
if __name__ == "__main__":
    main()