import glob
import os
from typing import List

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms
from transformers import AutoModel, AutoProcessor, AutoTokenizer

from eval.constants import CLIP_MODEL_NAME, DEVICE, DINO_MODEL_NAME


def collect_jpg_files(directory: str) -> List[str]:
    jpg_files = glob.glob(os.path.join(directory, "**", "*.jpg"), recursive=True)
    return jpg_files


def clip_image_metric(pred_imgs: List[Image.Image], ref_path: str) -> float:
    clip_processor = AutoProcessor.from_pretrained(CLIP_MODEL_NAME)
    clip_model = AutoModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)

    pred_img_processed = clip_processor(
        images=pred_imgs, return_tensors="pt"
    ).pixel_values.to(DEVICE)
    pred_features = clip_model.get_image_features(pred_img_processed).unsqueeze(
        1
    )  # shape: [n_pred, 1, feature_dim]

    ref_imgs = collect_jpg_files(ref_path)
    ref_imgs = [Image.open(img) for img in ref_imgs]
    ref_imgs_processed = clip_processor(
        images=ref_imgs, return_tensors="pt"
    ).pixel_values.to(DEVICE)
    ref_features = clip_model.get_image_features(ref_imgs_processed).unsqueeze(
        0
    )  # shape: [1, n_ref, feature_dim]

    cosine_similarities = F.cosine_similarity(
        pred_features, ref_features, dim=-1
    )  # shape: [n_pred, n_ref]

    mean_similarities = cosine_similarities.mean(dim=1)  # shape: [n_pred]
    overall_mean_similarity = mean_similarities.mean().item()

    return overall_mean_similarity


def dino_metric(pred_imgs: List[Image.Image], ref_path: str) -> float:
    dino_model = AutoModel.from_pretrained(DINO_MODEL_NAME, add_pooling_layer=False).to(
        DEVICE
    )

    T = transforms.Compose(
        [
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    pred_imgs_processed = torch.stack([T(img).to(DEVICE) for img in pred_imgs])
    pred_features = dino_model(pred_imgs_processed).last_hidden_state[
        :, 0, :
    ]  # shape: [n_pred, feature_dim]

    ref_imgs = collect_jpg_files(ref_path)
    ref_imgs_processed = torch.stack(
        [T(Image.open(img)).to(DEVICE) for img in ref_imgs]
    )
    ref_features = dino_model(ref_imgs_processed).last_hidden_state[
        :, 0, :
    ]  # shape: [n_ref, feature_dim]

    pred_features = pred_features.unsqueeze(1)  # shape: [n_pred, 1, feature_dim]
    ref_features = ref_features.unsqueeze(0)  # shape: [1, n_ref, feature_dim]

    cosine_similarities = F.cosine_similarity(pred_features, ref_features, dim=-1)
    mean_similarities = cosine_similarities.mean(dim=1)  # shape: [n_pred]

    overall_mean_similarity = mean_similarities.mean().item()

    return overall_mean_similarity


def clip_text_metric(pred_imgs: List[Image.Image], ref_texts: List[str]) -> float:
    clip_processor = AutoProcessor.from_pretrained(CLIP_MODEL_NAME)
    clip_tokenizer = AutoTokenizer.from_pretrained(CLIP_MODEL_NAME)
    clip_model = AutoModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)

    processed_imgs = clip_processor(
        images=pred_imgs, return_tensors="pt"
    ).pixel_values.to(DEVICE)
    pred_features = clip_model.get_image_features(
        processed_imgs
    )  # shape: [n_pred, feature_dim]

    processed_texts = clip_tokenizer(
        text=ref_texts, return_tensors="pt", padding=True, truncation=True
    ).input_ids.to(DEVICE)
    ref_features = clip_model.get_text_features(
        processed_texts
    )  # shape: [n_ref, feature_dim]

    pred_features = pred_features.unsqueeze(1)  # shape: [n_pred, 1, feature_dim]
    ref_features = ref_features.unsqueeze(0)  # shape: [1, n_ref, feature_dim]
    sim = F.cosine_similarity(
        pred_features, ref_features, dim=-1
    )  # shape: [n_pred, n_ref]

    mean_sim = sim.mean(dim=1)  # shape: [n_pred]
    overall_mean_sim = mean_sim.mean().item()

    return overall_mean_sim


def calculate_cl_metrics(metric_matrix):
    T = metric_matrix.shape[0]

    avg_accuracies = []
    for i in range(T):
        valid_accuracies = [
            metric_matrix[i, j]
            for j in range(i + 1)
            if not np.isnan(metric_matrix[i, j])
        ]
        if valid_accuracies:
            avg_accuracies.append(np.mean(valid_accuracies))
    final_avg_accuracy = np.mean(avg_accuracies)

    forgetting_values = []
    for j in range(T):
        task_accuracies = [
            metric_matrix[i, j] for i in range(T) if not np.isnan(metric_matrix[i, j])
        ]
        if len(task_accuracies) > 1:
            initial_accuracy = task_accuracies[0]
            max_forgetting = max(initial_accuracy - acc for acc in task_accuracies[1:])
            forgetting_values.append(max_forgetting)
    final_avg_forgetting = np.mean(forgetting_values) if forgetting_values else 0.0

    return final_avg_accuracy, final_avg_forgetting
