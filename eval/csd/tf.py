import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

np.random.seed(0)

_size = 224
_normalize = transforms.Normalize(
    (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
)

transforms_branch0 = transforms.Compose(
    [
        transforms.Resize(size=_size, interpolation=F.InterpolationMode.BICUBIC),
        transforms.CenterCrop(_size),
        transforms.ToTensor(),
        _normalize,
    ]
)
