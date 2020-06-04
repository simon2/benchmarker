import torch


def get_kernel(params, unparsed_args):
    precision = "fp32"
    repo = "NVIDIA/DeepLearningExamples:torchhub"
    Net = torch.hub.load(repo, "nvidia_ssd", model_math=precision)
    return Net
