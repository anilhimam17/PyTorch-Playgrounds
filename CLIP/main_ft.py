import torch
from transformers import AutoProcessor, AutoModel


class PreTrainedCLIP:

    hf_model_id = "openai/clip-vit-base-patch32"

    def __init__(self) -> None:
        self.preprocessor = AutoProcessor.from_pretrained(PreTrainedCLIP.hf_model_id)
        self.base_model = AutoModel.from_pretrained(PreTrainedCLIP.hf_model_id)

    def __call__(self, image: torch.Tensor, text: list[str]) -> torch.Tensor:
        preprocessed_inputs = self.preprocessor(text=text, image=image, return_tensors="pt", padding=True)
        outputs = self.base_model(**preprocessed_inputs)

        return outputs
    

# The Main Function
def main():
    clip = PreTrainedCLIP()
    print("Model Architecture:\n", clip.base_model, end="\n\n---------------------------\n\n")

    print("Total No of Parameters: ", sum(layer.numel() for layer in clip.base_model.parameters()))


if __name__ == "__main__":
    main()