import torch
import torchvision

cpu_device = torch.device("cpu")
gpu_device = torch.device("cuda")


def eval_object_detection_model(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    n_examples: int,
) -> list[dict[str, torch.Tensor]]:
    model.eval()
    model.to(gpu_device)
    images = [dataset[i][0] for i in range(n_examples)]
    gpu_images = [img.to(gpu_device) for img in images]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    outputs = model(gpu_images)
    nms_outputs = []
    for output in outputs:
        keep_indices = torchvision.ops.nms(output["boxes"], output["scores"], 0.2)
        nms_outputs.append(
            {x: output[x][keep_indices] for x in ["boxes", "labels", "scores"]},
        )
    return [{k: v.to(cpu_device) for k, v in t.items()} for t in nms_outputs]
