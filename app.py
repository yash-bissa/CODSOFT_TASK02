from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from model import METRIC_NAME, MODEL_NAME, predict, train

app = FastAPI()
ROOT = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def trained_model():
    return train()


def load_model():
    try:
        return trained_model()
    except (FileNotFoundError, ValueError, KeyError) as error:
        raise HTTPException(
            503, f"Check the dataset files listed in README.md. {error}"
        ) from error


@app.get("/")
def home():
    return FileResponse(ROOT / "index.html")


@app.get("/api/metrics")
def metrics():
    _, scores = load_model()
    return {"model": MODEL_NAME, "f1_label": METRIC_NAME, **scores}


@app.post("/api/predict")
def prediction(payload: dict):
    model, _ = load_model()
    try:
        return predict(model, payload.get("input"))
    except (ValueError, AttributeError) as error:
        raise HTTPException(422, str(error)) from error


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8001)
