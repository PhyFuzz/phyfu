import torch

from phyfu.array_utils.rand_interface import RandUtils

torch.random.manual_seed(0)


class TorchRandUtils(RandUtils):
    @staticmethod
    def randint(low, high, size=None):
        if size is None:
            return torch.randint(low, high, size=(1,)).item()
        return torch.randint(low, high, size)

    @staticmethod
    def rand_normal(mean, sigma, size):
        return torch.randn(size) * sigma + mean