from transformers import (
    CLIPConfig,
    CLIPTextConfig,
    CLIPVisionConfig,
    CLIPModel,
    CLIPProcessor,
    CLIPImageProcessorFast,
    CLIPTokenizerFast
)
from peft import (
    LoraConfig,
    PeftModel,
    PeftMixedModel,
    get_peft_model
)
import torch


class PreTrainedCLIP:

    hf_model_id = "openai/clip-vit-base-patch16"

    def __init__(self, lora_config: LoraConfig) -> None:

        # ==== Instance Variables ====
        self.lora_config = lora_config

        # ==== Configuring and Compiling the Base Model ====

        # Configuration to be used by the Text Encoder defaults to Paper Specs
        text_config = CLIPTextConfig()

        # Configuration to be used by the Image Encoder defaults to ViT-Base/32 => Enabling ViT-Base/16
        vision_config = CLIPVisionConfig(patch_size=16, num_hidden_layers=12)

        # Compiled Model Configuration
        self.clip_config = CLIPConfig(
            text_config=text_config.to_dict(),
            vision_config=vision_config.to_dict()
        )

        # Compiled Base Model
        self.base_model = CLIPModel.from_pretrained(PreTrainedCLIP.hf_model_id, config=self.clip_config)

        # ==== Configuring and Compiling the Preprocessor ====

        # Configuration to be used for the Text Tokenizer
        tokenizer_config = CLIPTokenizerFast.from_pretrained(PreTrainedCLIP.hf_model_id)

        # Configuration to be used for the Image Processor
        vision_processor_config = CLIPImageProcessorFast.from_pretrained(PreTrainedCLIP.hf_model_id)

        # Compiled Proprocessor Configuration
        self.preprocessor = CLIPProcessor(
            image_processor=vision_processor_config, 
            tokenizer=tokenizer_config
        )

        # Preparing the base model with LoRA Adapters
        self.peft_model = self.add_lora_adapters()

    def __call__(self, image: torch.Tensor, text: list[str]) -> torch.Tensor:
        """Special Method for the forward propagation of the PretrainedCLIP model 
        based on Custom Dataset."""

        preprocessed_inputs = self.preprocessor(
            text=text, 
            images=image, 
            return_tensors="pt",    # type: ignore | Pylance doesn't track kwargs
            padding=True            # type: ignore | Pylance doesn't track kwargs
        )
        outputs = self.peft_model(**preprocessed_inputs)

        return outputs
    
    def add_lora_adapters(self) -> PeftModel | PeftMixedModel:
        """Helper function to add the LoRA Adaptors to the pre-trained model."""

        return get_peft_model(
            model=self.base_model,
            peft_config=self.lora_config
        )
