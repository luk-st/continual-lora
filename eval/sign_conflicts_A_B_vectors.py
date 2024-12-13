import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from diffusers import AutoencoderKL, StableDiffusionXLPipeline

RESULTS_DIR = "./results-conflicts-objects"

BASE_VAE_PATH = "madebyollin/sdxl-vae-fp16-fix"
BASE_SDXL_PATH = "stabilityai/stable-diffusion-xl-base-1.0"


def load_pipe_from_model_task(model_path, method_name, device):
    if model_path == "stabilityai/stable-diffusion-xl-base-1.0":
        pipeline = StableDiffusionXLPipeline.from_pretrained(model_path, torch_dtype=torch.float16)

    elif method_name in ["merge_and_init", "ortho_init", "mag_max_light"]:
        vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", torch_dtype=torch.float16)
        pipeline = StableDiffusionXLPipeline.from_pretrained(model_path, vae=vae, torch_dtype=torch.float16)

    elif method_name in ["naive_cl"]:
        vae = AutoencoderKL.from_pretrained(BASE_VAE_PATH, torch_dtype=torch.float16)
        pipeline = StableDiffusionXLPipeline.from_pretrained(BASE_SDXL_PATH, vae=vae, torch_dtype=torch.float16)
        pipeline.load_lora_weights(model_path)
        pipeline.fuse_lora(fuse_unet=True)
        pipeline.unload_lora_weights()

    pipeline = pipeline.to(device)
    return pipeline


def get_tasks(file_path):
    with open(file_path) as f:
        file = json.load(f)
    return file["tasks"]


def get_order_seed(file_path):
    return int(file_path.split("/")[-1].split("seed_")[1].split("_")[0])


def get_seed_seed(file_path):
    return int(file_path.split("/")[-2].split("seed_")[1].split("_")[0])


def _get_models(method_name, task_type, seed, order):
    models_path = f"./models/{method_name}/seed_{seed}_{task_type}/seed_{order}_order"
    tasks = get_tasks(file_path=(Path(models_path) / Path("config.json")).resolve())
    n_tasks = len(tasks)
    # TODO: as always problem with style
    return [f"{models_path}/{i}" for i in range(1, n_tasks + 1)]


def _get_subtract_models(models_to_subtract):
    return ["stabilityai/stable-diffusion-xl-base-1.0"] + models_to_subtract[:-1]


def _get_params_number_of_layer(layer):
    return layer.shape.numel()


def _get_number_of_sign_conflicts(vector_A, vector_B):
    return torch.sum(vector_A.cpu() * vector_B.cpu() < 0).item()


def _get_unet_att_weights(model):
    att_weights = {
        k: v.cpu()
        for k, v in model.state_dict().items()
        if any(sub in k for sub in ["to_w", "to_k", "to_out", "to_v"])
    }
    assert len(att_weights) == 140 * 4
    return att_weights


def get_vector_differs(pipe, pipe_to_subtract):
    main_vector = _get_unet_att_weights(pipe.components["unet"])
    subtract_vector = _get_unet_att_weights(pipe_to_subtract.components["unet"])
    return {k: v - subtract_vector[k] for k, v in main_vector.items()}


def update_final_df(df, sign_conflicts_dict, main_model):
    main_name = main_model.split("/")[-1]
    for model_B, sign_conflicts in sign_conflicts_dict.items():
        model_B_name = model_B.split("/")[-1]
        df.at[model_B_name, main_name] = (
            sign_conflicts
            if pd.isna(df.at[model_B_name, main_name])
            else df.at[model_B_name, main_name] + sign_conflicts
        )


def calculate_sign_conflicts(model_A, model_to_subtract, models_to_compare, method_name):
    vector_A = get_vector_differs(model_A, model_to_subtract)
    sign_conflicts_dict_norm, sign_conflicts_dict = {}, {}

    for model_B_path in models_to_compare:
        model_B = load_pipe_from_model_task(model_path=model_B_path, method_name=method_name, device="cuda")
        vector_B = get_vector_differs(model_B, model_to_subtract)

        # calculate using number of sign conflicts
        sign_conflicts = sum(
            _get_number_of_sign_conflicts(layer_A, layer_B)
            for layer_A, layer_B in zip(vector_A.values(), vector_B.values())
        )
        all_params = sum(_get_params_number_of_layer(layer_A) for layer_A in vector_A.values())
        sign_conflicts_dict[model_B_path] = (sign_conflicts / all_params) * 100

        # calculate using norm of vectors
        for layer in vector_A:
            sign_conflict = np.sign(vector_A[layer]) != np.sign(vector_B[layer])
            vector_B[layer] = np.where(sign_conflict, vector_B[layer], 0)

        norm_vector1 = np.linalg.norm(np.concatenate([val.flatten() for val in vector_A.values()]))
        norm_vector2 = np.linalg.norm(np.concatenate([val.flatten() for val in vector_B.values()]))
        sign_conflicts_dict_norm[model_B_path] = norm_vector2 / norm_vector1

        del model_B

    return sign_conflicts_dict, sign_conflicts_dict_norm


def main(models_path, task_type, method_name):
    seed_seed = get_seed_seed(models_path)
    order_seed = get_order_seed(models_path)

    out_path = (Path(RESULTS_DIR) / Path(f"{task_type}/{method_name}/order_{order_seed}/seed_{seed_seed}")).resolve()
    os.makedirs(out_path, exist_ok=True)

    model_names = get_tasks(file_path=(Path(models_path) / Path("config.json")).resolve())
    # TODO: as always problem with style
    model_names = [str(i) for i in range(1, len(model_names) + 1)]
    final_df = pd.DataFrame(index=model_names, columns=model_names)
    final_df_norm = pd.DataFrame(index=model_names, columns=model_names)

    print(final_df)

    models = _get_models(method_name=method_name, task_type=task_type, seed_seed=seed_seed, order_seed=order_seed)
    models_to_subtract = _get_subtract_models(models_to_subtract=models)

    print(f"Seed: {seed_seed}")
    print(f"Models: {models}")
    print(f"Models to subtract: {models_to_subtract}")

    for idx, (main_model, model_to_subtract) in enumerate(zip(models[:-1], models_to_subtract)):
        print(
            f"Calculating sign conflicts: \n\tMain model: {main_model}\n\tModel to subtract: {model_to_subtract}\n\tModels to compare: {models[idx+1:]}"
        )

        model_A = load_pipe_from_model_task(model_path=main_model, method_name=method_name, device="cuda")
        model_to_subtract = load_pipe_from_model_task(
            model_path=model_to_subtract, method_name=method_name, device="cuda"
        )

        sign_conflicts_dict, sign_conflicts_dict_norm = calculate_sign_conflicts(
            model_A, model_to_subtract, models[idx + 1 :], method_name
        )

        update_final_df(df=final_df, sign_conflicts_dict=sign_conflicts_dict, main_model=main_model)
        update_final_df(df=final_df_norm, sign_conflicts_dict=sign_conflicts_dict_norm, main_model=main_model)

        del model_A

    final_df.to_csv(os.path.join(out_path, f"sign_conflicts_avg.csv"))
    final_df_norm.to_csv(os.path.join(out_path, f"sign_conflicts_avg_norm.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_path", type=str, required=True)
    parser.add_argument(
        "--method_name", type=str, required=True, choices=["mag_max_light", "merge_and_init", "naive_cl", "ortho_init"]
    )
    parser.add_argument("--task_type", choices=["style", "object"], required=True, type=str, help="Type of experiment")
    args = parser.parse_args()

    main(task_type=args.task_type, models_path=args.models_path, method_name=args.method_name)
