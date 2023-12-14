import os
import cv2
import torch
import numpy as np
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data.detection_utils import read_image
from densepose.vis.densepose_results import DensePoseResultsFineSegmentationVisualizer
from densepose.vis.extractor import DensePoseResultExtractor
from densepose import add_densepose_config
from segment import segment_mask
import warnings
from omegaconf import OmegaConf
from cldm.model import create_model
from utils import tensor2img
from cldm.plms_hacked import PLMSSampler

warnings.simplefilter("ignore")

# Set up Detectron2 config
cfg = get_cfg()
add_densepose_config(cfg)
cfg.merge_from_file("configs/densepose.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # Confidence threshold for detection
cfg.MODEL.WEIGHTS = "ckpts/model_final_162be9.pkl"  # "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl"
cfg.freeze()

IMG_H = 512
IMG_W = 384
CONFIG_PATH = "configs/VITON512.yaml"
MODEL_LOAD_PATH = "ckpts/VITONHD.ckpt"
SAVE_DIR = "image"
DENOISE_STEPS = 50
ETA = 0.0

config = OmegaConf.load(CONFIG_PATH)
config.model.params.img_H = IMG_H
config.model.params.img_W = IMG_W
params = config.model.params

model = create_model(config_path=None, config=config)
model.load_state_dict(torch.load(MODEL_LOAD_PATH, map_location="cpu"))
model = model.cuda()
model.eval()

sampler = PLMSSampler(model)

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
    H, W, _ = image.shape

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
    mask = cloth_mask # (mask + cloth_mask).astype(np.uint8)
    mask[mask != 0] = 255
    # Taking a matrix of size 5 as the kernel
    k_size = (int(H * 0.05), int(W * 0.05))
    kernel = np.ones(k_size, np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    mask_path = get_output_path(image_path, prefix="mask")
    agn_path = get_output_path(image_path, prefix="agnostic")
    cv2.imwrite(mask_path, mask)

    image = image.copy()
    inverted_mask = cv2.bitwise_not(mask.copy())
    image[(inverted_mask == 0)] = 128
    cv2.imwrite(agn_path, image)
    return dense_path, mask_path, agn_path


def imread(path, h=IMG_H, w=IMG_W, is_mask=False, in_inverse_mask=False, img=None):
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    if img is None:
        img = cv2.imread(path)
    if not is_mask:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (w, h))
        img = (img.astype(np.float32) / 127.5) - 1.0  # [-1, 1]
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (w, h))
        img = (img >= 128).astype(np.float32)  # 0 or 1
        img = img[:, :, None]
        if in_inverse_mask:
            img = 1 - img
    return img


@torch.no_grad()
def tryon(
    img_fn,
    cloth_fn,
    model=model,
    sampler=sampler,
    params=params,
):
    img_fp = os.path.basename(img_fn)
    cloth_fp = os.path.basename(cloth_fn)
    _, ext = os.path.splitext(img_fp)
    to_path = os.path.join(SAVE_DIR, img_fp.replace(ext, f"_{cloth_fp}"))

    shape = (4, IMG_H // 8, IMG_W // 8)
    save_dir = SAVE_DIR
    os.makedirs(save_dir, exist_ok=True)

    densepose_fn, mask_fn, agn_fn = preprocess(img_fn)
    image = imread(img_fn)
    cloth = imread(cloth_fn)
    agn = imread(agn_fn)
    agn_mask = imread(mask_fn, is_mask=True, in_inverse_mask=True)
    image_densepose = imread(densepose_fn)

    batch = {
        "agn": torch.unsqueeze(torch.from_numpy(agn), dim=0),
        "agn_mask": torch.unsqueeze(torch.from_numpy(agn_mask), dim=0),
        "cloth": torch.unsqueeze(torch.from_numpy(cloth), dim=0),
        "image": torch.unsqueeze(torch.from_numpy(image), dim=0),
        "image_densepose": torch.unsqueeze(torch.from_numpy(image_densepose), dim=0),
        "txt": [""],
        "img_fn": [img_fn],
        "cloth_fn": [cloth_fn],
    }

    z, c = model.get_input(batch, params.first_stage_key)
    bs = z.shape[0]
    c_crossattn = c["c_crossattn"][0][:bs]
    if c_crossattn.ndim == 4:
        c_crossattn = model.get_learned_conditioning(c_crossattn)
        c["c_crossattn"] = [c_crossattn]
    uc_cross = model.get_unconditional_conditioning(bs)
    uc_full = {"c_concat": c["c_concat"], "c_crossattn": [uc_cross]}
    uc_full["first_stage_cond"] = c["first_stage_cond"]
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.cuda()
    sampler.model.batch = batch

    ts = torch.full((1,), 999, device=z.device, dtype=torch.long)
    start_code = model.q_sample(z, ts)

    samples, _, _ = sampler.sample(
        DENOISE_STEPS,
        bs,
        shape,
        c,
        x_T=start_code,
        verbose=False,
        eta=ETA,
        unconditional_conditioning=uc_full,
    )

    x_samples = model.decode_first_stage(samples)
    for sample_idx, (x_sample, fn, cloth_fn) in enumerate(
        zip(x_samples, batch["img_fn"], batch["cloth_fn"])
    ):
        x_sample_img = tensor2img(x_sample)  # [0, 255]
        repaint_agn_img = np.uint8(
            (batch["image"][sample_idx].cpu().numpy() + 1) / 2 * 255
        )  # [0,255]
        repaint_agn_mask_img = batch["agn_mask"][sample_idx].cpu().numpy()  # 0 or 1
        x_sample_img = repaint_agn_img * repaint_agn_mask_img + x_sample_img * (
            1 - repaint_agn_mask_img
        )
        x_sample_img = np.uint8(x_sample_img)
        print(f"Try On results saved to: {to_path}")
        cv2.imwrite(to_path, x_sample_img[:, :, ::-1])

    torch.cuda.empty_cache()
    return to_path
