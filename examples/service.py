import pathlib

import torch
from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel

app = FastAPI()

loaded_models = {}
for model_file in list(pathlib.Path().glob("*.pkl")):
    logger.info(f"Loading {model_file}.")
    loaded_models[model_file.stem] = torch.load(
        model_file,
        map_location=torch.device("cpu"),
    )
logger.info("All models loaded.")


class ObjectDetectionInferenceRequest(BaseModel):
    name: str
    image: bytes


@app.get("/models")
async def models() -> list[str]:
    return list(loaded_models)


@app.post("/detect")
async def detect(request: ObjectDetectionInferenceRequest) -> bool:
    return True
