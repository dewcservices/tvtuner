import pathlib

import torch
from torchvision import utils

colors = ["black", "red", "yellow", "white"]


def export_boxes(
    images: torch.Tensor,
    boxes: torch.Tensor,
    labels: torch.Tensor,
    label_names: list[str],
    image_suffix: str,
    output_dir: pathlib.Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, (image, boxes_, labels_) in enumerate(
        zip(images, boxes, labels, strict=False),
    ):
        example = utils.draw_bounding_boxes(
            (255 * image).type(torch.uint8),
            boxes_,
            [label_names[j] for j in labels_],
            width=5,
            colors=[colors[j] for j in labels_],
        )
        utils.save_image(example.float() / 255, output_dir / f"{i}-{image_suffix}.jpg")
