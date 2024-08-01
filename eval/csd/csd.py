import torch
from torch.nn import functional as F

from .model import CSD_CLIP
from .tf import transforms_branch0
from .utils import convert_state_dict

CHECKPOINT_PATH = "res/csd_checkpoint.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_csd_model(device=DEVICE) -> CSD_CLIP:
    csd_model = CSD_CLIP()
    checkpoint = torch.load(CHECKPOINT_PATH)
    state_dict = convert_state_dict(checkpoint["model_state_dict"])
    msg = csd_model.load_state_dict(state_dict, strict=False)
    csd_model = csd_model.to(device)
    return csd_model


def get_img(image, device=DEVICE):
    image_tensor = transforms_branch0(image.convert("RGB"))
    image_tensor = image_tensor.unsqueeze(dim=0).to(device)
    return image_tensor


def csdimage_metric(pred_imgs, ref_imgs, csd_model) -> float:
    pred_imgs = [get_img(img) for img in pred_imgs]
    style_output_pred_list = []
    for pred_img in pred_imgs:
        _, _, style_output_pred = csd_model(pred_img)
        style_output_pred_list.append(style_output_pred.cpu().detach())

    pred_img_tensors = torch.cat(style_output_pred_list, dim=0).unsqueeze(1)

    ref_imgs = [get_img(img) for img in ref_imgs]
    style_output_ref_list = []
    for ref_img in ref_imgs:
        _, _, style_output_ref = csd_model(ref_img)
        style_output_ref_list.append(style_output_ref.cpu().detach())

    ref_img_tensors = torch.cat(style_output_ref_list, dim=0).unsqueeze(0)

    cosine_similarities = F.cosine_similarity(
        pred_img_tensors, ref_img_tensors, dim=-1
    )  # shape: [n_pred, n_ref]

    mean_similarities = cosine_similarities.mean(dim=1)  # shape: [n_pred]
    overall_mean_similarity = mean_similarities.mean().item()

    return overall_mean_similarity
