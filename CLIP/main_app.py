from pathlib import Path
from torchvision.io import decode_image

from CLIP.src.load_model import FineTunedModel


# Relative Path to access the assets
ASSET_PATH = Path("CLIP/assets")


# The Main Function
def main() -> None:
    
    # Instantiating the Loaded FineTuned Model
    ft_model = FineTunedModel()

    # Loading the images
    bike = decode_image(str(ASSET_PATH / "bike.jpg")).unsqueeze(dim=0)
    car = decode_image(str(ASSET_PATH / "car.jpg")).unsqueeze(dim=0)
    aomine = decode_image(str(ASSET_PATH / "aomine.jpg")).unsqueeze(dim=0)
    naruto_jiraya = decode_image(str(ASSET_PATH / "naruto_jiraya.jpg")).unsqueeze(dim=0)
    
    # Creating the image embeddings
    bike_embedding = ft_model.generate_image_embedding(bike)
    car_embedding = ft_model.generate_image_embedding(car)
    aomine_embedding = ft_model.generate_image_embedding(aomine)
    naruto_jiraya_embedding = ft_model.generate_image_embedding(naruto_jiraya)

    # Text Input
    text_ls = ["ice popsicle."]
    
    # Creating the text embedding
    text_embedding = ft_model.generate_text_embedding(text_ls)

    # Norm Embeddings
    bike_embedding_norm = bike_embedding / bike_embedding.norm(dim=-1, keepdim=True)
    car_embedding_norm = car_embedding / car_embedding.norm(dim=-1, keepdim=True)
    aomine_embedding_norm = aomine_embedding / aomine_embedding.norm(dim=-1, keepdim=True)
    naruto_jiraya_embedding_norm = naruto_jiraya_embedding / naruto_jiraya_embedding.norm(dim=-1, keepdim=True)
    text_embedding_norm = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

    # Temperature Param
    logits_scale = ft_model.peft_model.base_model.model.logit_scale.exp()

    # Similarity Calculation
    bike_sim = (bike_embedding_norm @ text_embedding_norm.T) * logits_scale
    car_sim = (car_embedding_norm @ text_embedding_norm.T) * logits_scale
    aomine_sim = (aomine_embedding_norm @ text_embedding_norm.T) * logits_scale
    naruto_jiraya_sim = (naruto_jiraya_embedding_norm @ text_embedding_norm.T) * logits_scale

    print(f"Bike Sim: {bike_sim.item():.4f}")
    print(f"Car Sim: {car_sim.item():.4f}")
    print(f"Aomine Sim: {aomine_sim.item():.4f}")
    print(f"Naruto Sim: {naruto_jiraya_sim.item():.4f}")


# Driver Code
if __name__ == "__main__":
    main()