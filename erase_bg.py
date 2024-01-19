from skimage import io
import numpy as np
import requests


def rgba2rgb(rgba, background=(255, 255, 255)):
    row, col, ch = rgba.shape

    if ch == 3:
        return rgba

    assert ch == 4, "RGBA image has 4 channels."

    rgb = np.zeros((row, col, 3), dtype="float32")
    # r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]
    r, g, b, a = [rgba[:, :, c] for c in range(ch)]
    a = np.asarray(a, dtype="float32") / 255.0

    if not isinstance(background, str):
        R, G, B = background
    else:
        background = cv2.imread(background)
        background = cv2.resize(background, (col, row))
        R, G, B = background[:, :, 0], background[:, :, 1], background[:, :, 2]

    rgb[:, :, 0] = r * a + (1.0 - a) * R
    rgb[:, :, 1] = g * a + (1.0 - a) * G
    rgb[:, :, 2] = b * a + (1.0 - a) * B
    return np.asarray(rgb, dtype="uint8")


def erase_bg(img_path):
    url = "https://api.pixelbinz0.de/nagato/api/v2/core/erasebg"
    headers = {
        "accept": "application/json",
    }
    files = {
        "image_file": (img_path.split("/")[-1], open(img_path, "rb"), "image/jpeg")
    }

    data = {
        "image_url": "",
        "auto_scale": "true",
        "quality_type": "original",
        "industry_type": "general",
        "refine": "true",
        "shadow": "false",
        "matting_type": "fba-matting",
    }
    response = requests.post(url, headers=headers, files=files, data=data)
    img = io.imread(response.text)
    img = rgba2rgb(img)
    io.imsave(img_path, img)


if __name__ == "__main__":
    img_path = "image/8c30d5ea.png"
    erase_bg(img_path)
