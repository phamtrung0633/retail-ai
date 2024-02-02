from torchvision import transforms
from PIL import Image

import numpy as np
import torch

preprocess = transforms.Compose([
    transforms.Resize((224, 224), interpolation = 3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def LATransformerForward(model, device, imgs):
    with torch.no_grad():
        pilimgs = [preprocess(Image.fromarray(img)) for img in imgs]
        prepped = torch.stack(pilimgs).to(device)

        out = model(prepped).cpu()

        fnorm = torch.norm(out, p = 2, dim = 1, keepdim = True) * np.sqrt(14)
        qnorm = out.div(fnorm.expand_as(out))

        return qnorm.view((-1)) # 14 * 768 -> 10752
