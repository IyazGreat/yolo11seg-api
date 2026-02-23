from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from app.model import predict_image

app = FastAPI(title="YOLOv11-Seg API")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        result = predict_image(contents)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )