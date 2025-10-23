from pathlib import Path

import gradio as gr
import torch
from gradio.themes import Ocean
from torchvision.io import ImageReadMode, decode_image

from CLIP.src.load_model import (
    MILD_SIMILARITY_OFFSET,
    SIMILARITY_THREASHOLD,
    FineTunedModel,
)


class UserInterface:
    """This class implements the gradio interface for the AI Photo Search Engine."""

    def __init__(self) -> None:
        
        # Placeholder for any long descriptions and styling for the website.
        self.block_params = {"title": "Research Companion", "fill_height": True, "fill_width": True, "theme": Ocean()}
        self.header_description = """
        <div class="description_style">
        Wanna leverage AI to search / organize your folders of photos rapidly
        <mark class="mark_style">Project Lighthouse</mark> is the way to go.<br>
        <mark class="mark_style">Upload a folder of images</mark> and search 
        for specific images you are looking for through
        <mark class="mark_style">Natural Language Text Prompts</mark>.
        </div>
        """
        self.file_css = """
        #upload_widget .file-preview-holder {max-height: 250px; overflow-y: auto;}
        #result_gallery {height: 30rem; min_height: 400px;}
        .description_style {text-align: center; line-height: 2.2; font-size: 16px; width: 100%;}
        .mark_style {background-color: #c9c8c7; padding: 0.2em 0.4em; border-radius: 5px;}
        .header_style {text-align: center; font-size: 32px; margin-top: 10px; width: 100%;}
        """

        # Creating an instance of the Loaded FineTuned Model.
        self.ft_model = FineTunedModel()

        # Retrieving the FineTuned Temperature Parameter.
        self.temperature = self.ft_model.peft_model.base_model.model.logit_scale.exp()

    def main_page(self) -> None:
        """Loads all the UI elements for the main page."""

        # Loading the Properties for the Block Interface Layout.
        with gr.Blocks(**self.block_params, css=self.file_css) as main_page_demo:
            
            # A registry for all the image embeddings.
            image_embedding_index: gr.State = gr.State({})

            # Main Title as a Div Row
            with gr.Row():
                _ = gr.Markdown(
                    "<H1 class='header_style'>Project Lighthouse 🗼</H1>"
                )
            # Main Description as Div Row
            with gr.Row():
                _ = gr.Markdown(self.header_description)

            # Main Div to partition the interface
            with gr.Row():

                # Left Column
                with gr.Column(scale=1):
                    file_widget = gr.File(label="Upload Images", file_count="directory", elem_id="upload_widget")

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

                    # Clear and Reset
                    clear_and_reset_btn = gr.Button("Clear and Reset")

                # Right Column
                with gr.Column(scale=1):
                    image_display_gallery = gr.Gallery(
                        label="Top Hits",
                        file_types=["image"],
                        rows=1,
                        columns=1,
                        object_fit="contain",
                        elem_id="result_gallery"
                    )

            # API Patches

            # The Entrypoint to the generate_image_embedding API
            file_widget.upload(
                fn=self.index_images,
                inputs=[file_widget, image_embedding_index],
                outputs=[top_n, text_box, image_embedding_index]
            )

            # The Entrypoint to the generate_text_embedding API
            text_box.submit(
                fn=self.find_text_image_hits,
                inputs=[text_box, top_n, image_embedding_index],
                outputs=image_display_gallery
            )

            # Clear and Reset the Application.
            clear_and_reset_btn.click(
                fn=self.clear_states,
                outputs=[file_widget, text_box, image_display_gallery, image_embedding_index]
            )

        # Running the Main Page.
        main_page_demo.launch()

    def index_images(self, root_image_path: list[str], image_embedding_index: dict[str, tuple[torch.Tensor, str]]):
        """Retrieves the images that were input to the File widget
        to generate the image embeddings using the Vision Encoder.
        
        args:
        - root_image_path: list[str] -> A list of absolute paths for all the uploaded 
        images in the gradio private backend directory.
        - image_embedding_index: dict[str, tuple[torch.Tensor, str]] -> The global session state to store
        the indexed images from upload for each user privately.

        returns:
        - tuple[
            - gr.Update: bool -> Set the interactive property of the top_n slider to true post indexing.
            - image_embedding_index: dict[str, tuple[torch.Tensor, str] -> Stores the names of the images 
            to avoid duplication with the corresponding generated image embeddings for lookup. The dictionary
            is tracked by gr.State() which maintains a global session state.
        ]
        """

        # Information for the user on beginning indexing
        gr.Info(message="Please Wait!!! I am learning the images just now.")

        # Updating the UI during indexing
        yield gr.update(interactive=False), gr.update(interactive=False), image_embedding_index

        # Root path to the private gradio backend.
        self.backend_path: str = str(Path(root_image_path[0]).parent)

        # Iterating through each of the uploaded images.
        for image_path in root_image_path:

            # Absolute Path to the Image
            path_handle = Path(image_path)

            # Preventing duplication of images based on Image Stem.
            if path_handle.stem in image_embedding_index.keys():
                continue
            else:
                # Loading the Image.
                image_data = decode_image(str(path_handle), mode=ImageReadMode.RGB).unsqueeze(dim=0)

                # Generating the Image Embedding
                image_embedding = self.ft_model.generate_image_embedding(image_data)
                image_embedding_norm = image_embedding / image_embedding.norm(dim=-1, keepdim=True)

                # Storing the Normalised Tensor into the Session State.
                image_embedding_index[path_handle.stem] = image_embedding_norm, image_path

        # Information to the user on completing indexing
        gr.Info(message="Image Learning Process completed, ready to search for images.")

        yield (
            gr.update(interactive=True),
            gr.update(interactive=True),
            image_embedding_index
        )
    
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
    
    def find_text_image_hits(self, text_prompt: str, top_n: float, image_embedding_index: dict[str, tuple[torch.Tensor, str]]) -> list[str]:
        """Orchestrates the image search by taking a text input generating its embeddings
        and utilising the embeddings to lookup the image_embedding_index to display to most similar image.
        
        args:
        - text_prompt: str -> A string description for the image to be searched.
        - image_embedding_index: dict[str, tuple[torch.Tensor, str]] -> The global session state to store
        the indexed images from upload for each user privately. 
        
        returns:
        - image_path: str -> Absolute path for the most similar image to be displayed."""

        # If the prompt entered is empty.
        if not text_prompt:
            gr.Warning(message="No Text Prompt provided, please input a valid text prompt before an image search.")
            return []
        
        # If no images were uploaded.
        if not image_embedding_index:
            gr.Warning(message="No Images uploaded, please upload images before an image search.")
            return []

        # Retrieving the Text Embedding.
        text_embedding_norm = self.embed_text(text_prompt=text_prompt)

        # Similarity Lookup
        sim_lookup: dict[str, torch.Tensor] = {}

        for image_name, image_embedding_norm_and_path in image_embedding_index.items():
            
            # Similarity Calculation
            sim_value = (image_embedding_norm_and_path[0] @ text_embedding_norm.T) * self.temperature
            sim_lookup[image_name] = sim_value

        # Identifying the highest hit.
        img_scores = sorted(
            list(sim_lookup.items()),
            key=lambda x: x[-1],
            reverse=True
        )

        # Retrieving the path for the top n hit images.
        top_n_hit_image_paths = [image_embedding_index[image_name][-1] for image_name, _ in img_scores[:int(top_n)]]

        # Warning to the user for less overlap.

        # Case 1. Mild Confidence based on the Similarity Scores.
        top_sim_score = img_scores[0][1]
        if top_sim_score.item() >= SIMILARITY_THREASHOLD and top_sim_score < (SIMILARITY_THREASHOLD + MILD_SIMILARITY_OFFSET):
            gr.Info("Mild Matches found, advise a sharper text prompt to improve results.")
        
        # Case 2. Very little Confidence based on the Similarity Scores.
        elif top_sim_score.item() < SIMILARITY_THREASHOLD:
            gr.Warning("No Strong Match with any of the learnt images, displayed images might not highlight the context.")

        return top_n_hit_image_paths
    
    def clear_states(self):
        """Clears the states for all the components."""

        return gr.update(value=None), gr.update(value=""), gr.update(value=None), {}
