import torch


__all__ = ('get_device',)


def get_device(device: torch.device | str = None):
    if type(device) == torch.device:
        return device
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda' 
        else:
            device = 'cpu'
    return torch.device(device)