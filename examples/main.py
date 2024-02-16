import pathlib
import sys

import torch
from dataset import get_dataset
import torchvision
from torchvision import utils

from tvtuner.model import get_model_object_detection
from tvtuner.train import train

dataset_name = sys.argv[1]  # penn_fudan, tomatoes, uav
dataset_train, dataset_test, num_classes, labels = get_dataset(dataset_name)

model = get_model_object_detection(num_classes)

output_dir = pathlib.Path(f"/tmp/outputs/{dataset_name}")
output_dir.mkdir(parents=True, exist_ok=True)
colors=["black", "red", "yellow", "white"]
for i in range(10):
    image, metadata = dataset_test[i]
    example = utils.draw_bounding_boxes(
        (255 * image).type(torch.uint8),
        metadata["boxes"],
        [labels[j] for j in metadata["labels"]],
        width=5,
        colors=[colors[j] for j in metadata["labels"]]
    )
    utils.save_image(example.float() / 255, output_dir / f"{i}-label.jpg")

for k in range(3):
    train(model, dataset_train, dataset_test, 1)
 
    cpu_device = torch.device("cpu")
    gpu_device = torch.device("cuda")
    model.eval()
    model.to(gpu_device)
    images = [dataset_test[i][0] for i in range(10)]
    gpu_images = list(img.to(gpu_device) for img in images)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    outputs = model(gpu_images)
    nms_outputs = []
    for output in outputs:
        keep_indices = torchvision.ops.nms(output["boxes"], output["scores"], 0.2)
        nms_outputs.append({
            x: output[x][keep_indices] for x in ["boxes", "labels", "scores"]
        })
    cpu_outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in nms_outputs]

    for i in range(10):
        image = images[i]
        example = utils.draw_bounding_boxes(
            (255 * image).type(torch.uint8),
            cpu_outputs[i]["boxes"],
            [labels[j] for j in cpu_outputs[i]["labels"]],
            width=5,
            colors=[colors[j] for j in cpu_outputs[i]["labels"]]
        )
        utils.save_image(example.float() / 255, output_dir / f"{i}-{k}-trained.jpg")
