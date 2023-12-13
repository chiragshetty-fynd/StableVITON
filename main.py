from fastapi import FastAPI, HTTPException, Form

import os
from os.path import join as opj
from omegaconf import OmegaConf

import cv2
import torch
import numpy as np

from utils import tensor2img
from cldm.model import create_model
from cldm.plms_hacked import PLMSSampler

from preprocess import preprocess

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

app = FastAPI()

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
):
    img_fp = os.path.basename(img_fn)
    cloth_fp = os.path.basename(cloth_fn)
    _, ext = os.path.splitext(img_fp)
    to_path = os.path.join(SAVE_DIR, img_fp.replace(ext, f"_{cloth_fp}"))

    sampler = PLMSSampler(model)
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

    return to_path



@app.post("/tryon")
async def virtual_tryon(
    img_path: str = Form(...),
    cloth_path: str = Form(...),
):
    try:
        img_path = img_path.strip()
        cloth_path = cloth_path.strip()
        img_cloth_path = tryon(img_path, cloth_path)
        return {"generated_image": img_cloth_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8888)
