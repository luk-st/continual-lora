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

    _, _, pred_img_tensors = csd_model(torch.cat(pred_imgs, dim=0))
    pred_img_tensors = pred_img_tensors.cpu().detach().unsqueeze(1)

    ref_imgs = [get_img(img) for img in ref_imgs]
    _, _, ref_img_tensors = csd_model(torch.cat(ref_imgs, dim=0))
    ref_img_tensors = ref_img_tensors.cpu().detach().unsqueeze(0)

    cosine_similarities = F.cosine_similarity(
        pred_img_tensors, ref_img_tensors, dim=-1
    )  # shape: [n_pred, n_ref]

    mean_similarities = cosine_similarities.mean(dim=1)  # shape: [n_pred]
    overall_mean_similarity = mean_similarities.mean().item()

    return overall_mean_similarity
