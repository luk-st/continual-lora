import os
import argparse

import pandas as pd
import numpy as np
import torch
from diffusers import DiffusionPipeline

STYLE_MODELS = [
    "watercolor_painting_style_sd1",
    "oil_painting_style_sd2",
    "flat_cartoon_illustration_sd3",
    "abstract_rainbow_colored_flowing_smoke_wave_sd4",
    "sticker_sd5",
]
OBJECT_MODELS = [
    "wolf_plushie_sd1",
    "backpack_sd2",
    "dog6_sd3",
    "candle_sd4",
    "cat2_sd5",
]

SEEDS = [0, 5, 10, 15]
FINAL_RESULTS_PATH = "./results/sign_conflicts_avaraged"


def _get_final_results_path(style):
    return f"{FINAL_RESULTS_PATH}/{'style' if style else 'object'}_sign_conflicts_avaraged"


def _get_models(seed, style=True):
    model_type = "style" if style else "object"
    models = STYLE_MODELS if style else OBJECT_MODELS
    return [f"./models/seed_{seed}_{model_type}/{model_name}" for model_name in models]


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


def calculate_sign_conflicts(model_A, model_to_subtract, models_to_compare):
    vector_A = get_vector_differs(model_A, model_to_subtract)
    sign_conflicts_dict_norm, sign_conflicts_dict = {}, {}

    for model_B_path in models_to_compare:
        model_B = DiffusionPipeline.from_pretrained(model_B_path, torch_dtype=torch.float16)
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


def main(style=True):
    os.makedirs(_get_final_results_path(style), exist_ok=True)

    model_names = STYLE_MODELS if style else OBJECT_MODELS
    final_df = pd.DataFrame(index=model_names, columns=model_names)
    final_df_norm = pd.DataFrame(index=model_names, columns=model_names)

    for seed in SEEDS:
        models = _get_models(seed, style)
        models_to_subtract = _get_subtract_models(models)

        print(f"Seed: {seed}")
        print(f"Models: {models}")
        print(f"Models to subtract: {models_to_subtract}")

        for idx, (main_model, model_to_subtract) in enumerate(zip(models[:-1], models_to_subtract)):
            print(
                f"Calculating sign conflicts: \n\tMain model: {main_model}\n\tModel to subtract: {model_to_subtract}\n\tModels to compare: {models[idx+1:]}"
            )

            model_A = DiffusionPipeline.from_pretrained(main_model, torch_dtype=torch.float16)
            model_to_subtract = DiffusionPipeline.from_pretrained(model_to_subtract, torch_dtype=torch.float16)

            sign_conflicts_dict, sign_conflicts_dict_norm = calculate_sign_conflicts(
                model_A, model_to_subtract, models[idx + 1:]
            )

            update_final_df(final_df, sign_conflicts_dict, main_model)
            update_final_df(final_df_norm, sign_conflicts_dict_norm, main_model)

            del model_A

    final_df /= len(SEEDS)
    final_df_norm /= len(SEEDS)

    final_df.to_csv(os.path.join(FINAL_RESULTS_PATH, f"sign_conflicts_avg.csv"))
    final_df_norm.to_csv(os.path.join(FINAL_RESULTS_PATH, f"sign_conflicts_avg_norm.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--style", action="store_true", help="Calculate for style models")
    args = parser.parse_args()

    main(style=args.style)
