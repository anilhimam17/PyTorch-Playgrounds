from src.data import DataHandler


# The Main Function
def main() -> None:
    print("Hello, CLIP")
    
    # Loading the Data Handler
    data_handle = DataHandler()

    # Constructing the dataset
    data_handle.construct_dataset(batch_size=1024)


# Driver code
if __name__ == "__main__":

    # The Main Function
    main()
