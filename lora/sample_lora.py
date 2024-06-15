import argparse
import os

import torch
from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline
from huggingface_hub.repocard import RepoCard


def main(args):
    model_path = args.pretrained_model_name_or_path
    # card = RepoCard.load(lora_model_id)
    # base_model_id = card.data.to_dict()["base_model"]

    diffusion_pipe = DiffusionPipeline.from_pretrained(
        model_path, torch_dtype=torch.float16
    )
    if args.use_cuda:
        print("Using CUDA...")
        diffusion_pipe = diffusion_pipe.to("cuda")
    # diffusion_pipe.load_lora_weights(lora_model_id)

    if args.use_refiner:
        print("Using refiner...")
        refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )
        if args.use_cuda:
            refiner = refiner.to("cuda")

    generator = torch.Generator("cuda") if args.use_cuda else torch.Generator("cpu")
    generator = generator.manual_seed(0)

    prompt = args.prompt

    output_type = "latent" if args.use_refiner else "pil"
    image = diffusion_pipe(
        prompt=prompt, output_type=output_type, generator=generator
    ).images[0]

    if args.use_refiner:
        image = refiner(
            prompt=prompt, image=image[None, :], generator=generator
        ).images[0]

    directory_path = os.path.dirname(args.save_path)

    os.makedirs(directory_path, exist_ok=True)

    image.save(args.save_path)


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA sampling script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--use_cuda",
        action="store_true",
        help="True if we want to use CUDA",
    )
    parser.add_argument(
        "--use_refiner",
        action="store_true",
        help="True if we want to use Refiner for better generations",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Prompt.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Output image path",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
