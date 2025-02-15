import argparse
import json
import math
import os
from pathlib import Path

import torch
from accelerate import PartialState
from accelerate.utils import gather_object
from diffusers import AutoencoderKL, StableDiffusionXLPipeline
from tqdm import tqdm

from eval.helpers import save_pickle

RESULTS_DIR = "./results"

N_SAMPLES_PER_PROMPT = 8
BATCH_SIZE = 5
N_STEPS = 50
GENERATOR_SEED = 42

BASE_VAE_PATH = "madebyollin/sdxl-vae-fp16-fix"
BASE_SDXL_PATH = "stabilityai/stable-diffusion-xl-base-1.0"


def get_prompt_templates(task_type):
    if task_type == "object":
        return [
            "a {} in a purple wizard outfit",
            "a {} in a police outfit",
            "a {} wearing a santa hat",
            "a {} in a jail",
            "a {} looking into a mirror",
        ]
    elif task_type == "style":
        return [
            "{} image of a wizard",
            "{} image of a policeman",
            "{} image of a santa hat",
            "{} image of a clock",
            "{} image of a mirror",
        ]
    else:
        raise ValueError(f"Unknown task type: {task_type}")


def get_order_seed(file_path):
    return int(file_path.split("/")[-1].split("seed_")[1].split("_")[0])


def get_seed_seed(file_path):
    return int(file_path.split("/")[-2].split("seed_")[1].split("_")[0])


def get_tasks(file_path, without_order=False):
    with open(file_path) as f:
        file = json.load(f)
    tasks = file["tasks"]

    if without_order:
        tasks_map = {int(idx) + 1: task for idx, task in enumerate(tasks)}
        for task_k, task_v in tasks_map.items():
            task_v["index"] = task_k
    else:
        tasks_map = {int(task["index"]): task for task in tasks}

    return tasks_map, len(tasks)


def get_device():
    distributed_state = PartialState()
    if distributed_state.is_main_process:
        print(f"> Multi-GPU state initialized")
    return distributed_state, distributed_state.device


def load_pipe_from_model_task(model_path, method_name, device):
    if method_name in ["merge_and_init", "mag_max_light", "ortho_init"]:
        vae = AutoencoderKL.from_pretrained(BASE_VAE_PATH, torch_dtype=torch.float32)
        pipeline = StableDiffusionXLPipeline.from_pretrained(model_path, vae=vae, torch_dtype=torch.float32)

    elif method_name in ["naive_cl"]:
        vae = AutoencoderKL.from_pretrained(BASE_VAE_PATH, torch_dtype=torch.float32)
        pipeline = StableDiffusionXLPipeline.from_pretrained(BASE_SDXL_PATH, vae=vae, torch_dtype=torch.float32)
        pipeline.load_lora_weights(model_path)

    elif method_name in ["base"]:
        vae = AutoencoderKL.from_pretrained(BASE_VAE_PATH, torch_dtype=torch.float32)
        pipeline = StableDiffusionXLPipeline.from_pretrained(BASE_SDXL_PATH, vae=vae, torch_dtype=torch.float32)

    pipeline = pipeline.to(device)
    return pipeline


def prepare_noise(device, n_prompts):
    def get_noise_per_prompt(device):
        generator = torch.Generator(device=device).manual_seed(GENERATOR_SEED)
        return torch.randn(
            (N_SAMPLES_PER_PROMPT, 4, 128, 128),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )

    return torch.cat([get_noise_per_prompt(device=device) for _ in range(n_prompts)])


def prepare_prompts(prompts_templates, task_token):
    prompts = []
    for prompt in prompts_templates:
        for _ in range(N_SAMPLES_PER_PROMPT):
            prompts.append(prompt.format(task_token))
    return prompts


def sample_ranked_batched(
    pipeline,
    prompts,
    noises,
    distributed_state: PartialState,
    device: torch.device,
    num_inference_steps=N_STEPS,
    batch_size=BATCH_SIZE,
    print_accelerate_status=False,
):
    generator = torch.Generator(device=device).manual_seed(GENERATOR_SEED)
    prompt_indices = list(range(len(prompts)))
    with distributed_state.split_between_processes(prompt_indices) as rank_indices:
        if print_accelerate_status:
            print(f"Split ({distributed_state.process_index}/{distributed_state.num_processes}): {rank_indices}")
        rank_prompt = [prompts[idx] for idx in rank_indices]
        rank_noises = noises[rank_indices]
        outputs = []
        loop = (
            tqdm(
                range(0, rank_noises.shape[0], batch_size),
                total=math.ceil(rank_noises.shape[0] / batch_size),
                desc="Batched sampling",
            )
            if distributed_state.is_local_main_process
            else range(0, rank_noises.shape[0], batch_size)
        )

        for idx_start in loop:
            idx_end = idx_start + batch_size
            images = pipeline(
                prompt=rank_prompt[idx_start:idx_end],
                num_inference_steps=num_inference_steps,
                generator=generator,
                latents=rank_noises[idx_start:idx_end],
            ).images
            outputs.append(images)
        outputs = sum(outputs, [])

    if distributed_state.is_local_main_process:
        print("Rank 0 finished, waiting for other ranks...")
    distributed_state.wait_for_everyone()
    all_outputs = gather_object(outputs)
    return all_outputs


def sample_cl_models(models_path, tasks_configs, method_name, out_path, prompt_templates, task_type):
    distributed_state, device = get_device()
    if distributed_state.is_main_process:
        print(f"> Starting: sampling")
    print_accelerate_status = True

    if method_name == "base":
        models_after_tasks = [list(tasks_configs.values())[sorted(list(tasks_configs.keys()))[-1] - 1]]
    else:
        models_after_tasks = list(tasks_configs.values())

    # loop over each model after task
    for model_after_task_config in tqdm(models_after_tasks, desc="Sampling models"):
        model_after_task_idx = model_after_task_config["index"]
        out_path_after_task = out_path / Path(f"after_task_{model_after_task_idx}")
        os.makedirs(out_path_after_task, exist_ok=True)

        model_path = (models_path / Path(f"{model_after_task_idx}")).resolve() if method_name != "base" else ""
        pipe: StableDiffusionXLPipeline = load_pipe_from_model_task(
            model_path,
            method_name,
            device=device,
        )
        pipe.set_progress_bar_config(disable=True)

        for task_number in tqdm(range(1, model_after_task_idx + 1), "Sampling on tasks"):
            task_config = tasks_configs[task_number]
            task_prompt = task_config["prompt"] if task_type == "object" else task_config["style"]

            noises = prepare_noise(device, n_prompts=len(prompt_templates))
            prompts = prepare_prompts(prompts_templates=prompt_templates, task_token=task_prompt)
            outputs = sample_ranked_batched(
                pipeline=pipe,
                prompts=prompts,
                noises=noises,
                distributed_state=distributed_state,
                device=device,
                print_accelerate_status=print_accelerate_status,
            )
            print_accelerate_status = False

            task_out_path = out_path_after_task
            save_pickle({"prompts": prompts, "samples": outputs}, task_out_path / Path(f"on_task_{task_number}.pkl"))


def make_all_dirs(out_path, n_tasks):
    os.makedirs(out_path, exist_ok=True)
    for task_idx in range(1, n_tasks + 1):
        os.makedirs(out_path / Path(f"after_task_{task_idx}"), exist_ok=True)


def get_object_metrics(models_path, method_name, task_type, config_path):
    if method_name == "base":
        out_path = (Path(RESULTS_DIR) / Path(f"{task_type}/base")).resolve()
        tasks, n_tasks = get_tasks(file_path=Path(config_path).resolve(), without_order=True)
    else:
        order_seed = get_order_seed(models_path)
        seed_seed = get_seed_seed(models_path)
        out_path = (
            Path(RESULTS_DIR) / Path(f"{task_type}/{method_name}/order_{order_seed}/seed_{seed_seed}")
        ).resolve()
        tasks, n_tasks = get_tasks(file_path=(Path(models_path) / Path("config.json")).resolve())

    prompt_templates = get_prompt_templates(task_type=task_type)

    make_all_dirs(out_path, n_tasks)

    sample_cl_models(
        models_path=models_path,
        tasks_configs=tasks,
        method_name=method_name,
        out_path=out_path,
        prompt_templates=prompt_templates,
        task_type=task_type,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_path", type=str, required=False, default=None)
    parser.add_argument(
        "--method_name",
        type=str,
        required=True,
        choices=["mag_max_light", "merge_and_init", "naive_cl", "ortho_init", "base"],
    )
    parser.add_argument("--config_path", type=str, required=False, default=None)
    parser.add_argument("--task_type", type=str, required=True, choices=["object", "style"])
    args = parser.parse_args()

    if args.method_name == "base" and args.config_path is None:
        raise ValueError("Base method requires config path")

    elif args.method_name != "base" and args.models_path is None:
        raise ValueError("Models path is required for non-base methods")

    get_object_metrics(
        models_path=args.models_path,
        method_name=args.method_name,
        task_type=args.task_type,
        config_path=args.config_path,
    )
