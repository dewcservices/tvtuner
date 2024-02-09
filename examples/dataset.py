import torch
from penn_fudan.dataset import PennFudanDataset
from yolo.dataset import YOLODataset

from tvtuner.utils import get_transform


def get_dataset(
    dataset_name: str,
) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, int]:
    if dataset_name == "penn_fudan":
        dataset_train = PennFudanDataset(
            "penn_fudan/data/PennFudanPed",
            get_transform(train=True),
        )
        dataset_test = PennFudanDataset(
            "penn_fudan/data/PennFudanPed",
            get_transform(train=False),
        )
        num_classes = 2
    elif dataset_name == "tomatoes":
        dataset_train = YOLODataset("yolo/tomatoes/train", get_transform(train=True))
        dataset_test = YOLODataset("yolo/tomatoes/val", get_transform(train=False))
        num_classes = 4
    elif dataset_name == "uav":
        dataset_train = YOLODataset("yolo/uav/train", get_transform(train=True))
        dataset_test = YOLODataset("yolo/uav/train", get_transform(train=False))
        num_classes = 2
    else:
        raise NotImplementedError

    return dataset_train, dataset_test, num_classes
