import os
import pathlib

import torch
from torchvision import tv_tensors
from torchvision.io import read_image
from torchvision.transforms.v2 import functional as F


class YOLODataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms) -> None:
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to ensure that they are aligned
        self.imgs = sorted(os.listdir(os.path.join(root, "images")))
        self.labels = sorted(os.listdir(os.path.join(root, "labels")))

    def __getitem__(self, idx: int) -> tuple:
        # load images and masks
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        labels_path = os.path.join(self.root, "labels", self.labels[idx])

        img_ref = ".".join(img_path.split("/")[-1].split(".")[:-1])
        labels_ref = ".".join(labels_path.split("/")[-1].split(".")[:-1])
        assert img_ref == labels_ref

        img = read_image(img_path)
        _, img_height, img_width = img.shape

        boxes = []
        labels = []
        with pathlib.Path(labels_path).open() as f:
            for line in f.readlines():
                cls, xc, yc, w, h = map(float, line.split())

                labels.append(1 + int(cls))  # +1 to include background
                boxes.append(
                    [
                        (xc - 0.5 * w) * img_width,
                        (yc - 0.5 * h) * img_height,
                        (xc + 0.5 * w) * img_width,
                        (yc + 0.5 * h) * img_height,
                    ],
                )

        num_objs = len(labels)
        boxes = torch.tensor(boxes)
        labels = torch.tensor(labels, dtype=torch.int64)

        image_id = idx

        area = 0 if num_objs == 0 else boxes[:, 2] * boxes[:, 3]

        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Wrap sample and targets into torchvision tv_tensors:
        img = tv_tensors.Image(img)

        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(
            boxes,
            format="XYXY",
            canvas_size=F.get_size(img),
        )
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self) -> int:
        return len(self.imgs)
