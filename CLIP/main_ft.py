from peft import LoraConfig

from CLIP.src.clip_lora import PreTrainedCLIP


# ==== Global Variables ====
LORA_CONFIG = LoraConfig(
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]
)

# The Main Function
def main():

    # Initialising the pretrained and adapted model
    clip = PreTrainedCLIP(lora_config=LORA_CONFIG)

    # Verifying the Architecture
    print("Model Architecture:\n", clip.peft_model, end="\n\n---------------------------\n\n")
    print("Total Base Trained Parameters: ", sum(layer.numel() for layer in clip.base_model.parameters()))
    print("Total LoRA Trainable Parameters: ", end="")
    clip.peft_model.print_trainable_parameters()


if __name__ == "__main__":
    main()