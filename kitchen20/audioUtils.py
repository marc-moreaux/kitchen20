import torch
from torch import nn
import random
import numpy as np


def _check_audio(tensor):
    if not isinstance(tensor, nn.Tensor):
        raise TypeError('tensor should be a torch tensor')
    if len(tensor.size()) > 2:
        raise TypeError('tensor representing audio should be at most ' +
                        '2Dimentional')

# Default data augmentation
class Pad(object):
    """Pad the given tensor on all sides with specified padding fill value

    Args:
        tensor (Tensor): Tensor of audio signal with shape (LxC)
        padding (int or tuple): Padding on each border. If a single int is provided this
            is used to pad all borders. If tuple of length 2 is provided this is the padding
            on left/right.
        fill: signal fill value for constant fill. 
            This value is only used when the padding_mode is constant

    Returns:
        tensor (Tensor): Tensor of audio signal with shape (CxL)
    """
    def __init__(self, padding, fill=0):
        self.padding = padding
        self.fill = fill

    def __call__(self, tensor):
        return nn.ConstantPad1d(pad, value)(tensor)
    

class RandomCrop(object):
    """Randomly crops a piece of tensor

    Args:
        size (int): size of the crop to retrieve
    
    Return:
        tensor (Tensor): signal tensor with shape (size, channels)

    """
    def __init__(self, size):
        self.size = size

    def __call__(self, tensor):
        orig_size = len(tensor)
        start = random.randint(0, org_size - self.size)
        return tensor[start: start + self.size]


# For strong data augmentation
class randomScale(object):
    """Randomly and artificially strech or shrink audio
    """
    def __init__(self, max_scale, interpolate='Linear'):
        self.max_scale = max_scale
        self.interpolate = interpolate

    def __call__(self, tensor):
        tensor = tensor.float()

        # Pick a new scale and generate list of scaled indexes
        scale = max_scale ** random.uniform(-1, 1)
        output_size = float(len(sound) * scale)
        ref = torch.arange(output_size) / scale

        # Select interpolation type
        if interpolate.lower() == 'linear':
            ref1 = ref.int()
            ref2 = np.minimum(ref1 + 1, len(sound) - 1)
            r = (ref - ref1).float()  # Ratio of sound[ref] to use
            scaled_sound = (sound[ref1.long()] * (1 - r) +
                            sound[ref2.long()] * r)
        elif interpolate.lower() == 'nearest':
            ref = ref.int()  # Nearest index
            scaled_sound = sound[ref.double()]
        else:
            raise Exception('Invalid interpolation mode {}'.format(
                interpolate))


torchaudio.transforms.Compose([
    Pad(10, fill=0),
    RandomCrop(16000),
])

