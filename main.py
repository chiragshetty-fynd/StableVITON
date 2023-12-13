from tryon import tryon
from fastapi import FastAPI, HTTPException, Form

app = FastAPI()


@app.post("/tryon")
async def virtual_tryon(
    img_path: str = Form(...),
    cloth_path: str = Form(...),
):
    try:
        img_cloth_path = tryon(img_path, cloth_path)
        return {"generated_image": img_cloth_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8888)
