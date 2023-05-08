import torch

from phyfu.array_utils.array_interface import ArrayUtils
from phyfu.array_utils.torch_random import TorchRandUtils

class TorchArrayUtils(ArrayUtils):

    @staticmethod
    def length(v):
        return torch.norm(v.flatten())

    @staticmethod
    def angle(v1, v2):
        u1 = v1.flatten() / TorchArrayUtils.length(v1)
        u2 = v2.flatten() / TorchArrayUtils.length(v2)
        return torch.arccos(torch.clip(torch.dot(u1, u2), -1.0, 1.0))

    @staticmethod
    def to_numpy(arr):
        return arr.detach().numpy()

    @staticmethod
    def tile(a, rep):
        return torch.tile(a, rep)

    @staticmethod
    def euc_dist(a1, a2):
        return torch.linalg.norm(a1 - a2)

    @staticmethod
    def zeros(shape):
        return torch.zeros(shape)

    @staticmethod
    def concatenate(arrays):
        return torch.concatenate(arrays)

    @property
    def random(self):
        return TorchRandUtils

    @staticmethod
    def loss_to_float(loss) -> float:
        if isinstance(loss, float):
            return loss
        return loss.item()

    @staticmethod
    def save(file, arr):
        torch.save(arr, file)

    @staticmethod
    def load(file):
        return torch.load(file)