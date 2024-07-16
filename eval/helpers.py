import itertools
import pickle

import torch
from diffusers import DiffusionPipeline

from eval.constants import BASE_SDXL_MODEL, DEVICE
from eval.metrics import clip_image_metric, dino_metric


def get_model_outs(
    pretrained_model_name_or_path: str,
    prompts: list,
    samples_per_prompt: int,
    device: str = DEVICE,
):
    model_path = pretrained_model_name_or_path
    diffusion_pipe = DiffusionPipeline.from_pretrained(
        model_path, torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    diffusion_pipe = diffusion_pipe.to(device)
    generator = torch.Generator(device)
    generator = generator.manual_seed(0)

    tasks_prompts = []
    tasks_samples = []
    for current_prompt in prompts:
        task_prompts = [current_prompt] * samples_per_prompt
        task_samples = diffusion_pipe(
            prompt=task_prompts, output_type="pil", generator=generator
        )
        task_prompts.append(task_prompts)
        tasks_samples.append(task_samples)

    return tasks_prompts, list(
        itertools.chain.from_iterable([out.images for out in tasks_samples])
    )


def sample_cl_models(
    models_paths,
    task_number,
    tasks_tokens,
    prompts_templates,
    n_tasks,
    samples_per_prompt=6,
    device: str = DEVICE,
):
    def sample_on_task(
        tasks_tokens, prompts_templates, model_path, samples_per_prompt, device
    ):
        task_outs = {}
        prompts = [
            prompt.format(tasks_tokens[curr_task_number + 1])
            for prompt in prompts_templates
        ]
        out_prompts, out_samples = get_model_outs(
            pretrained_model_name_or_path=model_path,
            prompts=prompts,
            samples_per_prompt=samples_per_prompt,
            device=device,
        )
        task_outs["prompts"] = out_prompts
        task_outs["samples"] = out_samples
        return task_outs

    per_task_outs = {}
    if task_number == 0:
        model_path = BASE_SDXL_MODEL
        tasks_limit = n_tasks
    else:
        model_path = models_paths[task_number]
        tasks_limit = task_number

    for curr_task_number in range(tasks_limit):
        per_task_outs[curr_task_number + 1] = sample_on_task(
            tasks_tokens=tasks_tokens,
            prompts_templates=prompts_templates,
            model_path=model_path,
            samples_per_prompt=samples_per_prompt,
            device=device,
        )
    return per_task_outs


def get_cl_lora_alignment_metrics(
    models_tasks_outputs, gt_datasets_paths, n_tasks, is_style
):
    def get_model_after_task_metrics_style(
        model_after_task_idx, num_tasks, models_tasks_outputs, gt_datasets_paths
    ):
        tasks_stats = {}
        for task in range(num_tasks):
            samples = models_tasks_outputs[model_after_task_idx][task + 1]["samples"]
            gt_path = gt_datasets_paths[task]
            tasks_stats[task + 1] = {
                "clip": clip_image_metric(samples, gt_path),
                "dino": dino_metric(samples, gt_path),
            }
        return tasks_stats

    def get_model_after_task_metrics_object(
        model_after_task_idx, num_tasks, models_tasks_outputs, gt_datasets_paths
    ):
        tasks_stats = {}
        for task in range(num_tasks):
            samples = models_tasks_outputs[model_after_task_idx][task + 1]["samples"]
            gt_path = gt_datasets_paths[task]
            tasks_stats[task + 1] = {
                "clip": clip_image_metric(samples, gt_path),
                "dino": dino_metric(samples, gt_path),
            }
        return tasks_stats

    metrics_foo = (
        get_model_after_task_metrics_style
        if is_style
        else get_model_after_task_metrics_object
    )

    models_stats = {}
    models_stats[0] = metrics_foo(
        model_after_task_idx=0,
        num_tasks=n_tasks,
        models_tasks_outputs=models_tasks_outputs,
        gt_datasets_paths=gt_datasets_paths,
    )

    for model_after_nr in range(1, n_tasks + 1):
        models_stats[model_after_nr] = metrics_foo(
            model_after_task_idx=model_after_nr,
            num_tasks=model_after_nr,
            models_tasks_outputs=models_tasks_outputs,
            gt_datasets_paths=gt_datasets_paths,
        )

    return models_stats


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)
