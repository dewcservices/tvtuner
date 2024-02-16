import pathlib

import torch
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

loaded_models = {}
for model_file in list(pathlib.Path().glob("*.pkl")):
    loaded_models[model_file.stem] = torch.load(model_file)


class ObjectDetectionInferenceRequest(BaseModel):
    name: str
    image: bytes


@app.get("/models")
async def models() -> list[str]:
    return list(loaded_models)


@app.post("/detect")
async def detect(request: ObjectDetectionInferenceRequest) -> bool:
    return True
