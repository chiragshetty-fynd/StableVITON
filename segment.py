import os
import cv2
import warnings
import numpy as np
import torch.nn as nn
from PIL import Image
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation

warnings.simplefilter("ignore")

processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
model = AutoModelForSemanticSegmentation.from_pretrained(
    "mattmdjaga/segformer_b2_clothes"
)
softmax = nn.Softmax(dim=1)


def get_mask_path(input_path):
    _, ext = os.path.splitext(input_path)
    return input_path.replace(ext, f"_mask{ext}")


def get_agn_path(input_path):
    _, ext = os.path.splitext(input_path)
    return input_path.replace(ext, f"_agnostic{ext}")


def segment_mask(image_path):
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")

    outputs = model(**inputs)
    logits = outputs.logits.cpu()

    upsampled_logits = nn.functional.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False,
    )

    x = softmax(upsampled_logits).argmax(dim=1)[0].detach().numpy()
    mask = (((x == 4) + (x == 7) + (x == 8) + (x == 17)) * 255.0).astype(np.uint8)
    return mask


if __name__ == "__main__":
    path = "image/00008_00.jpg"
    generate_masks(path)
