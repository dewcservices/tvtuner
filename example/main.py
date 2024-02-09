from dataset import PennFudanDataset, get_transform
from model import get_model_instance_segmentation

from tvtuner.train import train

dataset_train = PennFudanDataset("data/PennFudanPed", get_transform(train=True))
dataset_test = PennFudanDataset("data/PennFudanPed", get_transform(train=False))

num_classes = 2  # our dataset has two classes only - background and person
model = get_model_instance_segmentation(num_classes)

train(model, dataset_train, dataset_test)
