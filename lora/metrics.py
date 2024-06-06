from typing import List

import torch
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms
from transformers import AutoModel, AutoProcessor, AutoTokenizer

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
DINO_MODEL_NAME = "facebook/dino-vits16"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def clip_image_metric(pred_img_path: str, ref_img_paths: List[str]) -> float:
    clip_processor = AutoProcessor.from_pretrained(CLIP_MODEL_NAME)
    clip_model = AutoModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)

    pred_img = Image.open(pred_img_path)
    ref_imgs = [Image.open(img) for img in ref_img_paths]

    pred_img = clip_processor(images=pred_img, return_tensors="pt").to(DEVICE)
    pred_features = clip_model.get_image_features(**pred_img)

    ref_imgs = clip_processor(images=ref_imgs, return_tensors="pt").to(DEVICE)
    ref_features = clip_model.get_image_features(**ref_imgs)

    sim = F.cosine_similarity(pred_features, ref_features, dim=-1)
    return sim.mean().item()


def clip_text_metric(pred_img_path: str, ref_texts: List[str]) -> float:
    clip_processor = AutoProcessor.from_pretrained(CLIP_MODEL_NAME)
    clip_tokenizer = AutoTokenizer.from_pretrained(CLIP_MODEL_NAME)
    clip_model = AutoModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)

    pred_img = Image.open(pred_img_path)
    pred_img = clip_processor(images=pred_img, return_tensors="pt").to(DEVICE)
    pred_features = clip_model.get_image_features(**pred_img)

    ref_texts = clip_tokenizer(
        ref_texts, return_tensors="pt", padding=True, truncation=True
    ).to(DEVICE)
    ref_features = clip_model.get_text_features(**ref_texts)

    sim = F.cosine_similarity(pred_features, ref_features, dim=-1)
    return sim.mean().item()


def dino_metric(pred_img_path: str, ref_img_paths: List[str]) -> float:
    dino_model = AutoModel.from_pretrained(DINO_MODEL_NAME, add_pooling_layer=False).to(DEVICE)

    T = transforms.Compose(
        [
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    pred_img = T(Image.open(pred_img_path)).unsqueeze(0).to(DEVICE)
    pred_features = dino_model(pred_img).last_hidden_state

    ref_imgs = [Image.open(img) for img in ref_img_paths]
    ref_imgs = torch.stack([T(img) for img in ref_imgs]).to(DEVICE)
    ref_features = dino_model(ref_imgs).last_hidden_state

    sim = F.cosine_similarity(pred_features[:, 0], ref_features[:, 0], dim=-1)
    return sim.mean().item()
