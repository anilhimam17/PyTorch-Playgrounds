from src.data import IMDBMovieReview
from src.data_preprocess import DataPreprocessor
from src.optimization import OptimizationLoop
from src.my_gpt import MyGPT


# The Main Function
def main():
    # Downloading and Loading the Dataset
    imdb_review_dataset = IMDBMovieReview()

    # Filtering the columns of the Dataset
    review_string = imdb_review_dataset.refine_structure()

    # Loading the GPT Model Class
    gpt_model = MyGPT(imdb_review_dataset.vocab).to(device="mps")

    # Creating the data tensor
    data_preprocessor = DataPreprocessor(review_string, device="mps")
    data_preprocessor.create_data_tensor(gpt_model)
    
    # Splitting the data tensor
    train_set, valid_set, test_set = data_preprocessor.train_valid_test()

    # Creating the optimization loop
    optim_handle = OptimizationLoop(data_preprocessor, gpt_model, 1e-3)

    # Training the model
    optim_handle.train(50000, train_set, valid_set, 32, 8)

    del gpt_model


# Driver code
if __name__ == "__main__":
    main()
