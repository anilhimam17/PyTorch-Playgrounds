from src.data import DataHandler


# ==== Global Variable for Program Configuration ====
BATCH_SIZE = 64


# The Main Function
def main() -> None:
    print("Hello, CLIP")
    
    # Loading the Data Handler
    data_handle = DataHandler()

    # Viewing the sizes of the indices for Training and Validation Splits
    print("Len of Train Indices: ", data_handle.train_idx.shape)
    print("Len of Valid Indices: ", data_handle.valid_idx.shape)

    loaded_train, loaded_valid = data_handle.construct_dataset(batch_size=BATCH_SIZE)
    print("Len of Train Dataset: ", len(data_handle.train_dataset))
    print("Len of Valid Dataset: ", len(data_handle.valid_dataset))

    print("No of Batches in Train Loaded: ", len(loaded_train), " -> ", len(loaded_train) * BATCH_SIZE)
    print("No of Batches in Valid Loaded: ", len(loaded_valid), " -> ", len(loaded_valid) * BATCH_SIZE)


# Driver code
if __name__ == "__main__":

    # The Main Function
    main()
