from pathlib import Path

from CLIP.src.ui import UserInterface


# Relative Path to access the assets
ASSET_PATH = Path("CLIP/assets")


# The Main Function
def main() -> None:

    # Initialising the UI
    interface = UserInterface()

    # Launching the Demo
    interface.page()


# Driver Code
if __name__ == "__main__":
    main()