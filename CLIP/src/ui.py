import gradio as gr
from gradio.themes import Ocean

import torch
from torchvision.io import decode_image, ImageReadMode

from pathlib import Path

from CLIP.src.load_model import FineTunedModel


class UserInterface:
    """This class implements the gradio interface for the AI Photo Search Engine."""

    def __init__(self) -> None:
        
        # Placeholder for any long descriptions and styling for the website.
        self.block_params = {"title": "Research Companion", "fill_height": True, "fill_width": True, "theme": Ocean()}
        self.mark_style = 'background-color: #c9c8c7; padding: 0.2em 0.4em; border-radius: 5px;'
        self.desciption_style = 'text-align: center; line-height: 2.5; font-size: 16px;'
        self.header_description = f"""
        <div style="{self.desciption_style}">
        Wanna leverage AI to search / organize your folders of photos rapidly
        <mark style="{self.mark_style}">Project Lighthouse</mark> is the way to go.<br>
        <mark style="{self.mark_style}">Upload a folder of images</mark> and search 
        for specific images you are looking for through
        <mark style="{self.mark_style}">Natural Language Text Prompts</mark>.
        </div>
        """

        # Creating an instance of the Loaded FineTuned Model.
        self.ft_model = FineTunedModel()

        # Retrieving the FineTuned Temperature Parameter.
        self.temperature = self.ft_model.peft_model.base_model.model.logit_scale.exp()

        # A registry for all the image embeddings.
        self.image_embedding_index: dict[str, tuple[torch.Tensor, str]] = {}

    def page(self) -> None:
        """Loads all the UI elements for the page."""

        # Loading the Properties for the Block Interface Layout.
        with gr.Blocks(**self.block_params) as demo:

            # Main Title as a Div Row
            with gr.Row():
                _ = gr.Markdown(
                    "<H1 style='text-align: center; font-size: 32px; margin-top: 10px;'>Project Lighthouse 🗼</H1>"
                )
            # Main Description as Div Row
            with gr.Row():
                _ = gr.Markdown(self.header_description)

            # Main Div to partition the interface
            with gr.Row():

                # Left Column
                with gr.Column(scale=1):
                    file_widget = gr.File(label="Upload Images", file_count="directory")

                    # The textbox to search for images.
                    text_box = gr.Textbox(placeholder="Describe your image for search")

                    # The slider for top n images.
                    top_n = gr.Slider(
                        label="Top N Image Filter",
                        minimum=1,
                        maximum=10,
                        value=3,
                        step=1,
                        precision=0,
                        interactive=False
                    )

                # Right Column
                with gr.Column(scale=1):
                    status_box = gr.Label(label="Status", value="Upload Images for Indexing")
                    image_display_gallery = gr.Gallery(label="Top Hits", file_types=["image"], rows=1, columns=1)

            # API Patches

            # The Entrypoint to the generate_image_embedding API
            file_widget.upload(
                fn=self.index_images,
                inputs=file_widget,
                outputs=[status_box, top_n]
            )

            # The Entrypoint to the generate_text_embedding API
            text_box.submit(
                fn=self.find_text_image_hits,
                inputs=[text_box, top_n],
                outputs=image_display_gallery
            )

        # Running the Demo
        demo.launch()

    def index_images(self, root_image_path: list[str]):
        """Retrieves the images that were input to the File widget
        to generate the image embeddings using the Vision Encoder.
        
        args:
        - root_image_path: list[str] -> A list of absolute paths for all the uploaded 
        images in the gradio private backend directory.
        
        object_property:
        - image_embedding_index: dict[str, torch.Tensor] -> Stores the names of the images 
        to avoid duplication with the corresponding generated image embeddings for lookup.

        returns:
        - None
        """

        # Updating the UI during indexing
        yield "Indexing Images", gr.update(interactive=False)

        # Root path to the private gradio backend.
        self.backend_path: str = str(Path(root_image_path[0]).parent)

        # Iterating through each of the uploaded images.
        for image_path in root_image_path:

            # Creating a tensor from the raw image.
            path_handle = Path(image_path)
            image_data = decode_image(str(path_handle), mode=ImageReadMode.RGB).unsqueeze(dim=0)

            # Generating the Image Embedding and storing the normalised tensor.
            if path_handle.stem in self.image_embedding_index.keys():
                continue
            else:
                image_embedding = self.ft_model.generate_image_embedding(image_data)
                image_embedding_norm = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
                self.image_embedding_index[path_handle.stem] = image_embedding_norm, image_path

        yield f"Indexing Complete\nNo of Images Indexed: {len(self.image_embedding_index)}", gr.update(interactive=True)
    
    def embed_text(self, text_prompt: str) -> torch.Tensor:
        """Takes the input from the text prompt provided by the user and 
        generates the text embeddings using the Text Encoder.
        
        args:
        - text_prompt: str -> A string description for the image to be searched.
        
        returns:
        - text_embedding: torch.Tensor -> A contextually rich representation of the input
        text as a tensor.
        """

        # Generating the text embedding
        text_embedding = self.ft_model.generate_text_embedding([text_prompt])
        text_embedding_norm = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

        return text_embedding_norm
    
    def find_text_image_hits(self, text_prompt: str, top_n: float) -> list[str]:
        """Orchestrates the image search by taking a text input generating its embeddings
        and utilising the embeddings to lookup the image_embedding_index to display to most similar image.
        
        args:
        - text_prompt: str -> A string description for the image to be searched.
        
        returns:
        - image_path: str -> Absolute path for the most similar image to be displayed."""

        # If the prompt entered is empty.
        if not text_prompt:
            gr.Warning("Looks like the text prompt was empty, please provide a valid text prompt for image search.")
            return []

        # Retrieving the Text Embedding.
        text_embedding_norm = self.embed_text(text_prompt=text_prompt)

        # Similarity Lookup
        sim_lookup: dict[str, float] = {}

        for image_name, image_embedding_norm in self.image_embedding_index.items():
            
            # Similarity Calculation
            sim_value = (image_embedding_norm[0] @ text_embedding_norm.T) * self.temperature
            
            if image_name in sim_lookup.keys():
                continue
            else:
                sim_lookup[image_name] = sim_value

        # Identifying the highest hit.
        img_scores = sorted(
            list(sim_lookup.items()),
            key=lambda x: x[-1],
            reverse=True
        )

        # Retrieving the path for the top n hit images.
        top_n_hit_image_paths = [self.image_embedding_index[image_name][-1] for image_name, _ in img_scores[:int(top_n)]]

        return top_n_hit_image_paths
