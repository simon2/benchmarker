from pathlib import Path

import torch

from .pytorch_nvidia.model import SSD300


# from https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/SpeechSynthesis/Tacotron2/inference.py
def checkpoint_from_distributed(state_dict):
    """
    Checks whether checkpoint was generated by DistributedDataParallel. DDP
    wraps model in additional "module.", it needs to be unwrapped for single
    GPU inference.
    :param state_dict: model's state dict
    """
    ret = False
    for key, _ in state_dict.items():
        if key.find("module.") != -1:
            ret = True
            break
    return ret


# from https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/SpeechSynthesis/Tacotron2/inference.py
def unwrap_distributed(state_dict):
    """
    Unwraps model from DistributedDataParallel.
    DDP wraps model in additional "module.", it needs to be removed for single
    GPU inference.
    :param state_dict: model's state dict
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.1.", "")
        new_key = new_key.replace("module.", "")
        new_state_dict[new_key] = value
    return new_state_dict


def get_kernel(params, unparsed_args=None):
    ssd_cpu = SSD300()
    CACHE = Path("~/.cache/benchmarker/models/").expanduser()
    FNAME = "nvidia_ssdpyt_fp32_20190225.pt"
    PATH = CACHE.joinpath(FNAME)
    URL = "https://api.ngc.nvidia.com/"
    URL += "v2/models/nvidia/ssdpyt_fp32/versions/1/files/"
    URL += FNAME
    error_msg = "Download weights using wget {} -O {}".format(URL, PATH)
    assert PATH.exists(), error_msg
    ckpt = torch.load(PATH, map_location=lambda storage, loc: storage)
    ckpt = ckpt["model"]
    if checkpoint_from_distributed(ckpt):
        ckpt = unwrap_distributed(ckpt)
    ssd_cpu.load_state_dict(ckpt)
    return ssd_cpu
