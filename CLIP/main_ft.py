from CLIP.src.clip_lora import PreTrainedCLIP    


# The Main Function
def main():
    clip = PreTrainedCLIP()
    print("Model Architecture:\n", clip.base_model, end="\n\n---------------------------\n\n")

    print("Total No of Parameters: ", sum(layer.numel() for layer in clip.base_model.parameters()))


if __name__ == "__main__":
    main()