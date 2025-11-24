#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: Mar 6, 2022
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

# Standard Library Imports
import os
import random
import json
from pathlib import Path
import warnings

# Third-Party Libraries
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.functional import binary_cross_entropy_with_logits
import torchvision.transforms as T
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from pycocotools import mask as coco_mask
import cv2

# Project-Specific Modules
#from sam2.build_sam import build_sam2
#from sam2.sam2_image_predictor import SAM2ImagePredictor