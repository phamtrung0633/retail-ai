import torch
import numpy as np
from itertools import islice
from PIL import Image

from timm import create_model
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

NUM_CHANNELS = 3
MAX_GROUP = 10

def grouped(iterable, n):
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch

class VIT:

    def __init__(self, device):
        model = create_model(
            'swinv2_large_window12to24_192to384',
            pretrained = True,
            num_classes = 0
        )

        model.eval()

        if torch.cuda.is_available():
            model.cuda()

        self._transforms = create_transform(
            **resolve_data_config(model.pretrained_cfg, model = model)
        )

        self.device = device
        self._model = model
        self.dims = None
        self.prepare()

    def prepare(self):
        with torch.no_grad():
            dummy = Image.fromarray(np.zeros((1, 1, NUM_CHANNELS), dtype = np.uint8))
            batch = self._transforms(dummy).unsqueeze(0).to(self.device)

            self.dims = self._model(batch).squeeze().shape

    def forward(self, image):
        with torch.no_grad():
            pilimg = Image.fromarray(image)
            batch = self._transforms(pilimg).unsqueeze(0).to(self.device)

            return self._model(batch)

    def forward_many(self, images):
        with torch.no_grad():
            preprocessed = [self._transforms(Image.fromarray(image)) for image in images]
            outs = []

            for group in grouped(preprocessed, MAX_GROUP):
                batch = torch.stack(group).to(self.device)
                outs.append(self._model(batch))

            catted = torch.zeros(0).to(self.device)

            for out in outs:
                catted = torch.cat((catted, out), 0)

            return catted