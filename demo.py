import os
from os.path import join as opj
from omegaconf import OmegaConf
from importlib import import_module
import argparse

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from cldm.plms_hacked import PLMSSampler
from cldm.model import create_model
from utils import tensor2img


CONFIG_PATH = "configs/VITON512.yaml"
MODEL_LOAD_PATH = "ckpts/VITONHD.ckpt"
DATA_ROOT_DIR = "VITONHD_DATA"
REPAINT = True
UNPAIR = True
SAVE_DIR = "samples"
DENOISE_STEPS = 50
IMG_H = 512
IMG_W = 384
ETA = 0.0


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
def main(
    img_fn,
    cloth_fn,
    config_path=CONFIG_PATH,
    model_load_path=MODEL_LOAD_PATH,
    img_H=IMG_H,
    img_W=IMG_W,
    unpair=UNPAIR,
    denoise_steps=DENOISE_STEPS,
    save_dir=SAVE_DIR,
    repaint=REPAINT,
    eta=ETA,
):
    config = OmegaConf.load(config_path)
    config.model.params.img_H = img_H
    config.model.params.img_W = img_W
    params = config.model.params

    model = create_model(config_path=None, config=config)
    model.load_state_dict(torch.load(model_load_path, map_location="cpu"))
    model = model.cuda()
    model.eval()

    sampler = PLMSSampler(model)
    shape = (4, img_H // 8, img_W // 8)
    save_dir = opj(save_dir, "unpair" if unpair else "pair")
    os.makedirs(save_dir, exist_ok=True)

    image = imread(img_fn)
    cloth = imread(cloth_fn)
    agn = imread(img_fn.replace(".jpg", "_agnostic.jpg"))
    agn_mask = imread(
        img_fn.replace(".jpg", "_mask.png"), is_mask=True, in_inverse_mask=True
    )
    image_densepose = imread(img_fn.replace(".jpg", "_densepose.jpg"))

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

    for k, v in batch.items():
        if isinstance(v, list):
            print(f"{k}={v}")
        else:
            print(f"{k}={v.shape}")


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
        denoise_steps,
        bs,
        shape,
        c,
        x_T=start_code,
        verbose=False,
        eta=eta,
        unconditional_conditioning=uc_full,
    )

    x_samples = model.decode_first_stage(samples)
    for sample_idx, (x_sample, fn, cloth_fn) in enumerate(
        zip(x_samples, batch["img_fn"], batch["cloth_fn"])
    ):
        x_sample_img = tensor2img(x_sample)  # [0, 255]
        if repaint:
            repaint_agn_img = np.uint8(
                (batch["image"][sample_idx].cpu().numpy() + 1) / 2 * 255
            )  # [0,255]
            repaint_agn_mask_img = batch["agn_mask"][sample_idx].cpu().numpy()  # 0 or 1
            x_sample_img = repaint_agn_img * repaint_agn_mask_img + x_sample_img * (
                1 - repaint_agn_mask_img
            )
            x_sample_img = np.uint8(x_sample_img)

        img_fp = os.path.basename(img_fn)
        cloth_fp = os.path.basename(cloth_fn)
        to_path = os.path.join(save_dir, img_fp.replace('.jpg', f'_{cloth_fp}'))
        print(f'{to_path=} {x_sample_img.shape=}')
        cv2.imwrite(to_path, x_sample_img[:, :, ::-1])


if __name__ == "__main__":
    img_fn = "dataset/00008_00.jpg"
    cloth_fn = "dataset/00035_00.jpg"
    main(img_fn, cloth_fn)
