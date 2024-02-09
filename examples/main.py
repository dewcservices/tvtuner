from dataset import get_dataset

from tvtuner.model import get_model_object_detection
from tvtuner.train import train

dataset_name = "tomatoes"  # penn_fudan, tomatoes, uav
dataset_train, dataset_test, num_classes = get_dataset(dataset_name)
model = get_model_object_detection(num_classes)
train(model, dataset_train, dataset_test)
