import pathlib
import sys

from dataset import get_dataset

from tvtuner.eval import eval_object_detection_model
from tvtuner.model import get_model_object_detection
from tvtuner.train import train
from tvtuner.visuals import export_boxes

dataset_name = sys.argv[1]  # penn_fudan, tomatoes, uav
dataset_train, dataset_test, num_classes, label_names = get_dataset(dataset_name)

n_examples = 10
output_dir = pathlib.Path(f"/tmp/outputs/{dataset_name}")  # noqa: S108
export_boxes(
    [dataset_test[i][0] for i in range(n_examples)],
    [dataset_test[i][1]["boxes"] for i in range(n_examples)],
    [dataset_test[i][1]["labels"] for i in range(n_examples)],
    label_names,
    "label",
    output_dir,
)

model = get_model_object_detection(num_classes)

for k in range(3):
    train(model, dataset_train, dataset_test, 1)
    test_outputs = eval_object_detection_model(model, dataset_test, n_examples)
    export_boxes(
        [dataset_test[i][0] for i in range(n_examples)],
        [x["boxes"] for x in test_outputs],
        [x["labels"] for x in test_outputs],
        label_names,
        f"{k}-trained",
        output_dir,
    )
