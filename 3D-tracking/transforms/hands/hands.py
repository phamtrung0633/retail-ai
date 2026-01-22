from transforms import Transform, device
from transforms.hands.model import HandSegModel

import os

import torch
import torch.nn.functional as F

from torchvision import transforms

import numpy as np
import cv2

class HandSegmentor(Transform):
    _pool = None
    _count = 0

    def __init__(self, out_dims, map_size):
        super().__init__()

        self.resize = transforms.Resize(out_dims[::-1])
        self.out_dims = out_dims
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            self.resize,
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize(map_size[::-1])
        ])

    def is_initialised(self):
        return self.cls()._pool is not None

    def initialise(self):
        module = os.path.dirname(__file__)
        weights = 'weights/hands.ckpt'

        options = {'pretrained': True}

        model = HandSegModel.load_from_checkpoint(os.path.join(module, weights), **options)
        model.eval()

        model.to(device)

        self.set_pool(model)

    def forward(self, image):
        model = self.get_pool()
        sized = cv2.resize(image, self.out_dims)
        with torch.no_grad():
            batch = self.preprocess(sized).unsqueeze(0).to(device)
            logits = model(batch).cpu()
            preds = (F.softmax(logits, 1).argmax(1)[0] * 255).unsqueeze(0)
            mask = self.resize(preds).squeeze(0).numpy().astype(np.uint8)

        mask = cv2.bitwise_not(mask)
        return cv2.bitwise_and(sized, sized, mask = mask)