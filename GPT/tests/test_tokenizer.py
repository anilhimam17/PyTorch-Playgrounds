import pytest


# ==== Unit tests ====
def test_loading_model_works(model):
    """Tests if model instance was created by fixture."""
    assert model is not None, "There was an issue in loading the model."

@pytest.mark.parametrize(
        "input_seq, expected_token_seq",
        [
            ("Hello, World", [44, 73, 80, 80, 83, 16, 4, 59, 83, 86, 80, 72]),
            ("Now, I won't deny that wh", [50, 83, 91, 16, 4, 45, 4, 91, 83, 82, 11, 88, 4, 72, 73, 82, 93, 4, 88, 76, 69, 88, 4, 91, 76])
        ]
)
def test_encoder_works(model, input_seq, expected_token_seq):
    """Tests if model encoder provides accurate result."""
    assert model.encode(input_seq) == expected_token_seq, "Error in the encoding process of the tokenizer."

@pytest.mark.parametrize(
        "input_token_seq, expected_seq",
        [
            ([44, 73, 80, 80, 83, 16, 4, 59, 83, 86, 80, 72], "Hello, World"),
            ([50, 83, 91, 16, 4, 45, 4, 91, 83, 82, 11, 88, 4, 72, 73, 82, 93, 4, 88, 76, 69, 88, 4, 91, 76], "Now, I won't deny that wh")
        ]
)
def test_decoder_works(model, input_token_seq, expected_seq):
    """Tests if model decoder provides accurate result."""
    assert model.decode(input_token_seq) == expected_seq, "Error in the decoding process of the tokenizer."

@pytest.mark.parametrize(
    "input_string, output_string",
    [
        ("Hello, World", "Hello, World"),
        ("Bye, World", "Bye, World"),
        ("Hey GPT, Whatsup!!", "Hey GPT, Whatsup!!"),
        ("The Quick Brown Fox ... Little Dog!!", "The Quick Brown Fox ... Little Dog!!"),
    ]
)
def test_encoder_decoder_roundtrip(model, input_string, output_string):
    """Tests the complete tokenizer roundtrip."""
    assert model.decode(model.encode(input_string)) == output_string, "Tokenizer roundtrip failed."
