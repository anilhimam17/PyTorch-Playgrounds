from src.data import IMDBMovieReview
from src.my_gpt import MyGPT


# The Main Function
def main():
    # Downloading and Loading the Dataset
    imdb_review_dataset = IMDBMovieReview()

    # Filtering the columns of the Dataset
    review_string = imdb_review_dataset.refine_structure()

    # Data String Metrics
    imdb_review_dataset.datastring_metrics()

    # Loading the GPT Model Class
    gpt_model = MyGPT(imdb_review_dataset.vocab)

# Driver code
if __name__ == "__main__":
    main()
