import torch

from peft import PeftModel
from transformers import (
    CLIPConfig,
    CLIPTextConfig,
    CLIPVisionConfig,
    CLIPModel,
    CLIPImageProcessorFast,
    CLIPTokenizerFast
)


class FineTunedModel:
    """This class loads the fine tuned model for inferencing."""

    hf_base_model_id = "openai/clip-vit-base-patch16"
    hf_peft_model_id = "LaraYouThere17/Flicker30-CLIP-ViT-Base-16"

    def __init__(self, device: str = "mps") -> None:

        # ==== Loading the FineTuned Model ====

        # Configuring the Preprocessor for the Base Model
        self.tokenizer = CLIPTokenizerFast.from_pretrained(self.hf_base_model_id)
        self.vision_processor = CLIPImageProcessorFast.from_pretrained(self.hf_base_model_id)
        
        # Configuration for the Base Model
        text_config = CLIPTextConfig()
        vision_config = CLIPVisionConfig(patch_size=16, num_hidden_layers=12)
        base_config = CLIPConfig(
            text_config=text_config.to_dict(),
            vision_config=vision_config.to_dict()
        )

        # Loading the Pre-Trained Base Model
        self.base_model = CLIPModel.from_pretrained(
            pretrained_model_name_or_path=self.hf_base_model_id,
            config=base_config
        )

        # Loading the FineTuned Model
        self.peft_model = PeftModel.from_pretrained(
            self.base_model, 
            model_id=self.hf_peft_model_id
        ).to(device=device)
    
    def display_arch_and_params(self) -> None:
        """Displays the model architecture and params to console."""

        print("Model Architecture:\n", self.peft_model, end="\n\n---------------------------\n\n")
        print("Trained Params:")
        self.peft_model.print_trainable_parameters()

    def generate_text_embedding(self, text: list[str]) -> torch.Tensor:
        """Utilises the fine-tuned text encoder to generate a text embedding."""

        # Process the Text Input
        processed_text_inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        # Moving the Processed Input to GPU
        processed_text_inputs = {k: v.to(self.peft_model.device) for k, v in processed_text_inputs.items()}

        # Text Encoder Embedding Generation
        return self.peft_model.base_model.model.get_text_features(**processed_text_inputs)

    def generate_image_embedding(self, img: torch.Tensor) -> torch.Tensor:
        """Utilises the fine-tuned image encoder to generate an image embedding."""

        # Process the Image Input
        processed_image_inputs = self.vision_processor.preprocess(
            images=img,
            return_tensors="pt"
        )

        # Moving the Processed Input to GPU
        processed_image_inputs = {k: v.to(device=self.peft_model.device) for k, v in processed_image_inputs.items()}

        # Image Encoder Embedding Generation
        return self.peft_model.base_model.model.get_image_features(**processed_image_inputs)
