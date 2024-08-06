import torch
from diffusers import DiffusionPipeline


if __name__ == "__main__":
    pretrained_model = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
    )
    pretrained_state_dict = pretrained_model.unet.state_dict()

    # save the state dict
    torch.save(pretrained_state_dict, "res/state_dict_base.pth")
