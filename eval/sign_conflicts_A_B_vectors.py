import os

import pandas as pd
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
FINAL_RESULTS_PATH = "./results/sign_conflicts_avaraged_max"


def _get_results_dir_path(seed, style=True):
    return (
        f"./results/sign_conflicts_seed_{seed}_style"
        if style
        else f"./results/sign_conflicts_seed_{seed}_object"
    )


def _get_style_models(seed):
    return [f"./models/seed_{seed}_style/{model_name}" for model_name in STYLE_MODELS]


def _get_object_models(seed):
    return [f"./models/seed_{seed}_object/{model_name}" for model_name in OBJECT_MODELS]


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
    main_model_vector = _get_unet_att_weights(pipe.components["unet"])
    model_to_subtract_vector = _get_unet_att_weights(
        pipe_to_subtract.components["unet"]
    )
    vector_differs = {
        k: v - model_to_subtract_vector[k] for k, v in main_model_vector.items()
    }
    return vector_differs


def calculate_vector_A_B_signs_conflicts(model_A, model_to_subtract, models_to_compare):
    sign_conflicts_dict = {}

    vector_A = get_vector_differs(model_A, model_to_subtract)

    del model_A

    for model_B_path in models_to_compare:
        model_B = DiffusionPipeline.from_pretrained(
            model_B_path, torch_dtype=torch.float16
        )
        vector_B = get_vector_differs(model_B, model_to_subtract)

        sign_conflicts, all_params = 0, 0
        for layer_A, layer_B in zip(vector_A.values(), vector_B.values()):
            sign_conflicts += _get_number_of_sign_conflicts(layer_A, layer_B)
            all_params += _get_params_number_of_layer(layer_A)

        sign_conflicts_dict[model_B_path] = (sign_conflicts / all_params) * 100

        del model_B

    return sign_conflicts_dict


def main(style=True):
    final_df = pd.DataFrame(
        index=STYLE_MODELS if style else OBJECT_MODELS,
        columns=STYLE_MODELS if style else OBJECT_MODELS,
    )
    for seed in SEEDS:
        if style:
            models = _get_style_models(seed)
        else:
            models = _get_object_models(seed)

        df = pd.DataFrame(
            index=STYLE_MODELS if style else OBJECT_MODELS,
            columns=STYLE_MODELS if style else OBJECT_MODELS,
        )
        models_to_substract = _get_subtract_models(models)

        res_path = _get_results_dir_path(seed, style)

        print(f"Seed: {seed}")
        print(f"Models: {models}")
        print(f"Models to subtract: {models_to_substract}")

        for (idx, main_model), model_to_subtract in zip(
            enumerate(models[:-1]), models_to_substract
        ):
            print(
                f"Calculating sign conflicts: \n\tMain model: {main_model}\n\tModel to subtract: {model_to_subtract}\n\tModels to compare: {models[idx+1:]}"
            )
            model_A = DiffusionPipeline.from_pretrained(
                main_model, torch_dtype=torch.float16
            )
            model_to_subtract = DiffusionPipeline.from_pretrained(
                model_to_subtract, torch_dtype=torch.float16
            )
            sign_conflicts_dict = calculate_vector_A_B_signs_conflicts(
                model_A, model_to_subtract, models[idx + 1 :]
            )
            for model_B, sign_conflicts in sign_conflicts_dict.items():
                main_model = main_model.split("/")[-1]
                model_B = model_B.split("/")[-1]
                df.at[model_B, main_model] = sign_conflicts
                if (
                    pd.isna(final_df.at[model_B, main_model])
                    or final_df.at[model_B, main_model] == ""
                ):
                    final_df.at[model_B, main_model] = sign_conflicts
                else:
                    final_df.at[model_B, main_model] += sign_conflicts

            del model_A

        os.makedirs(res_path, exist_ok=True)
        df.to_csv(os.path.join(res_path, "sign_conflicts.csv"))

    os.makedirs(FINAL_RESULTS_PATH, exist_ok=True)
    final_df /= len(SEEDS)
    final_df.to_csv(os.path.join(FINAL_RESULTS_PATH, "sign_conflicts.csv"))


if __name__ == "__main__":
    main()
