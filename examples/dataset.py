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
        indices = torch.randperm(len(dataset_train)).tolist()
        dataset_train = torch.utils.data.Subset(dataset_train, indices[:-50])
        dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])
        num_classes = 2
        labels = ["background", "pedestrian"]
    elif dataset_name == "tomatoes":
        dataset_train = YOLODataset(
            "yolo/data/tomatoes/train", get_transform(train=True),
        )
        dataset_test = YOLODataset("yolo/data/tomatoes/val", get_transform(train=False))
        num_classes = 4
        labels = ["background", "unripe", "semi-ripe", "ripe"]
    elif dataset_name == "uav":
        dataset_train = YOLODataset("yolo/data/uav", get_transform(train=True))
        dataset_test = YOLODataset("yolo/data/uav", get_transform(train=False))
        indices = torch.randperm(len(dataset_train)).tolist()
        dataset_train = torch.utils.data.Subset(dataset_train, indices[:-50])
        dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])
        num_classes = 2
        labels = ["background", "uav"]
    else:
        raise NotImplementedError

    return dataset_train, dataset_test, num_classes, labels
