import os
import cv2
import numpy as np
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data.detection_utils import read_image
from densepose.vis.densepose_results import DensePoseResultsFineSegmentationVisualizer
from densepose.vis.extractor import DensePoseResultExtractor
from densepose import add_densepose_config
from segment import segment_mask
import warnings

warnings.simplefilter("ignore")

# Set up Detectron2 config
cfg = get_cfg()
add_densepose_config(cfg)
cfg.merge_from_file("configs/densepose.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # Confidence threshold for detection
cfg.MODEL.WEIGHTS = "ckpts/model_final_162be9.pkl"  # "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl"
cfg.freeze()

# Create Detectron2 predictor
predictor = DefaultPredictor(cfg)
extractor = DensePoseResultExtractor()
visualizer = DensePoseResultsFineSegmentationVisualizer(cfg)


def get_output_path(input_path, prefix="densepose"):
    _, ext = os.path.splitext(input_path)
    return input_path.replace(ext, f"_{prefix}{ext}")


def preprocess(image_path):
    # Read the input image
    image = read_image(image_path, format="BGR")
    blank = np.zeros_like(image)

    # Make prediction
    outputs = predictor(image)["instances"]

    # Extract the outputs
    outputs = extractor(outputs)

    # Visualize the outputs
    out = visualizer.visualize(blank, outputs)

    # Save the result
    dense_path = get_output_path(image_path)
    cv2.imwrite(dense_path, out)

    # 14, 32, 25, 42, 51, 56, 60, 67, 73, 77, 81, 101,
    mask = (
        # (out == 14) +
        (out == 32)
        + (out == 25)
        + (out == 42)
        + (out == 51)
        + (out == 56)
        +
        # (out == 60) +
        (out == 67)
        + (out == 73)
        + (out == 77)
        + (out == 101)
    )
    mask = ((mask != 0).astype(int) * 255.0).astype(np.uint8)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)[1]
    cloth_mask = segment_mask(image_path)
    mask = (mask + cloth_mask).astype(np.uint8)
    mask[mask != 0] = 255
    # Taking a matrix of size 5 as the kernel
    kernel = np.ones((50, 50), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    mask_path = get_output_path(image_path, prefix="mask")
    agn_path = get_output_path(image_path, prefix="agnostic")
    cv2.imwrite(mask_path, mask)

    image = image.copy()
    inverted_mask = cv2.bitwise_not(mask.copy())
    image[(inverted_mask == 0)] = 128
    cv2.imwrite(agn_path, image)
    return dense_path, mask_path, agn_path
