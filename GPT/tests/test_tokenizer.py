import pytest


# ==== Unit tests ====
def test_loading_model_works(model):
    """Tests if model instance was created by fixture."""
    assert model is not None, "There was an issue in loading the model."

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
