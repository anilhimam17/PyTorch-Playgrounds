import kagglehub
import pandas as pd

from pathlib import Path
import re
from typing import Generator


class IMDBMovieReview:
    """Class implements all the dataset loading, handling and metrics."""
    def __init__(self) -> None:
        self.dataset_path = Path(
            kagglehub.dataset_download(
                handle="atulanandjha/imdb-50k-movie-reviews-test-your-bert",
                force_download=False
            )
        )
        self.dataframe = pd.read_csv(
            filepath_or_buffer=self.dataset_path / "train.csv",
            encoding="utf-8"
        )

    def __iter__(self) -> Generator[str, str, None]:
        for i in range(10):
            yield self.dataframe["text"].loc[i]

    def refine_structure(self) -> str:
        """Focuses the entire dataframe to only the text removing other cols."""
        if "sentiment" in self.dataframe.columns:
            self.dataframe = self.dataframe.drop(["sentiment"], axis=1)
        self.dataset_string = "\n".join(self.dataframe["text"])
        
        # Cleaning up the string
        self.dataset_string = re.sub(r'[^\x00-\x7F]+', " ", self.dataset_string)
        self.dataset_string = re.sub(r'[\U00010000-\U0010ffff]+', " ", self.dataset_string)
        
        # Generating the vocabulary for the dataset string.
        self.vocab = sorted(list(set(self.dataset_string)))

        return self.dataset_string
    
    def datastring_metrics(self) -> None:
        """Provides metrics for the dataset string."""

        print(f"Length of the Dataset: {len(self.dataset_string)}\n")
        print(f"First 1000 Chars:\n{self.dataset_string[:1000]}\n")
        print(f"Vocabulary Size: {len(self.vocab)}\n")
        print(f"Vocabulary:\n{"".join(self.vocab)}")