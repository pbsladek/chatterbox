import logging

import torch


logger = logging.getLogger(__name__)


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def resolve_device(device):
    """Normalize device objects and indexed strings, retaining the MPS fallback."""
    device = torch.device(device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        logger.warning("MPS is unavailable; using CPU for inference.")
        return "cpu"
    return str(device)


def load_map_location(device):
    """Stage checkpoints on CPU; callers move tensors to the requested device.

    This also avoids restoring a CUDA checkpoint onto its original GPU index
    when the caller selected a different GPU.
    """
    torch.device(device)  # Validate strings and device objects before loading.
    return torch.device("cpu")
