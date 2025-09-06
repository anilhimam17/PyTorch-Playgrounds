from src.data import DataHandler


def main():
    """The main function for the Vision Transformer."""
    print("Hello, Vision Transformer")

    data_handle = DataHandler()

    # Training Set
    train_set = data_handle.load_set("train")
    class_names = train_set.classes

    print(f"Total No of Class: {len(class_names)}")
    print(f"Classes: {class_names}")
    print(f"Length of the Training-Dataset: {len(train_set)}")

    # Validation Set
    valid_set = data_handle.load_set("valid")
    class_names = valid_set.classes

    print(f"Length of the Validation-Dataset: {len(valid_set)}")


# ==== Driver Code ====
if __name__ == "__main__":
    main()