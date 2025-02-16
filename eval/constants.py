import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_SDXL_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
DINO_MODEL_NAME = "facebook/dino-vits16"
