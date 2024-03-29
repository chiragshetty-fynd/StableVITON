from process import tryon
from erase_bg import erase_bg
from fastapi import FastAPI, HTTPException, Form

app = FastAPI()


@app.post("/tryon")
async def virtual_tryon(
    img_path: str = Form(...),
    cloth_path: str = Form(...),
):
    try:
        img_path = img_path.strip()
        cloth_path = cloth_path.strip()
        erase_bg(img_path)
        erase_bg(cloth_path)
        img_cloth_path = tryon(img_path, cloth_path)
        return {"generated_image": img_cloth_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8888)
