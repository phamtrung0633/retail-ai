import time
from collections import defaultdict, OrderedDict
import json
import cv2

import torch

import numpy as np
from embeddings.embedder import Embedder
embedder = Embedder()
embedder.initialise()
shelf_id = "shelf_1"
image = cv2.imread("images/environmentLeft/1.png")
embedder.search(shelf_id, image)
# Config data
