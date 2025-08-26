class MyGPT:
    """Class implments the Generatively Pretrained Transformer from scratch using PyTorch."""
    def __init__(self, vocab: list[str] = []) -> None:
        self.encode_map = {ch: i for i, ch in enumerate(vocab)}
        self.decode_map = {i: ch for i, ch in enumerate(vocab)}

    def encode(self, input_string: str = "") -> list[int]:
        """Encode operation for the simple character level tokenizer."""
        return [self.encode_map[ch] for ch in input_string]

    def decode(self, input_seq: list[int] = []) -> str:
        """Invert operation for the simple character level tokenizer."""
        return "".join([self.decode_map[token] for token in input_seq])