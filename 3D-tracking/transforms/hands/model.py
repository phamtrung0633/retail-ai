import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models, transforms
from torch.optim import Adam

import pytorch_lightning as pl
import numpy as np

class HandSegModel(pl.LightningModule):
    """
    From: https://github.com/guglielmocamporese/hands-segmentation-pytorch
    This model is based on the PyTorch DeepLab model for semantic segmentation.
    """
    def __init__(self, pretrained, lr = 1e-4):
        super().__init__()

        self.deeplab = self._get_deeplab(pretrained = pretrained)
        self.lr = lr

    def _get_deeplab(self, pretrained=False, num_classes = 2):
        """
        Get the PyTorch DeepLab model architecture.
        """
        deeplab = models.segmentation.deeplabv3_resnet50(
            pretrained=False,
            num_classes=num_classes
        )

        if pretrained:
            deeplab_21 = models.segmentation.deeplabv3_resnet50(
                pretrained=True,
                progress=True,
                num_classes=21
            )
            for c1, c2 in zip(deeplab.children(), deeplab_21.children()):
                for p1, p2 in zip(c1.parameters(), c2.parameters()):
                    if p1.shape == p2.shape:
                        p1.data = p2.data

        return deeplab

    def forward(self, x):
        return self.deeplab(x)['out']

    def configure_optimizers(self):
        return Adam(self.parameters(), lr = self.lr)

def get_segmentor(*args, **kwargs):
    transform_dir = os.path.dirname(__file__)
    weights_path = 'weights/hands.ckpt'

    model_args = {'pretrained': True}
    model = HandSegModel.load_from_checkpoint(os.path.join(transform_dir, weights_path), **model_args)

    return model
