import numpy as np
from phyfu.array_utils.rand_interface import RandUtils


class NpRandUtils(RandUtils):
    @staticmethod
    def randint(low, high, size=None):
        return np.random.randint(low, high, size)

    @staticmethod
    def rand_normal(mean, sigma, size):
        return (np.random.randn(*size) * sigma + mean).astype(np.float32)
