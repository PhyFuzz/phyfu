import numpy as np

from phyfu.array_utils.array_interface import ArrayUtils
from phyfu.array_utils.np_random import NpRandUtils


class NpArrayUtils(ArrayUtils):
    @staticmethod
    def length(v):
        return np.linalg.norm(v.flatten())

    @staticmethod
    def angle(v1, v2):
        u1 = v1.flatten() / np.linalg.norm(v1.flatten())
        u2 = v2.flatten() / np.linalg.norm(v2.flatten())
        return np.arccos(np.clip(np.dot(u1, u2), -1.0, 1.0))

    @staticmethod
    def to_numpy(arr):
        return arr

    @staticmethod
    def tile(a, rep):
        return np.tile(a, reps=rep)

    @staticmethod
    def zeros(shape):
        return np.zeros(shape=shape, dtype=np.float32)

    @staticmethod
    def euc_dist(a1, a2):
        return np.linalg.norm(a1 - a2)

    @staticmethod
    def concatenate(arrays):
        return np.concatenate(arrays)

    @staticmethod
    def loss_to_float(loss) -> float:
        if isinstance(loss, float):
            return loss
        return loss.item()

    @staticmethod
    def save(file, arr):
        return np.savez(file, arr)

    @staticmethod
    def load(file):
        return np.load(file, allow_pickle=True)['arr_0']

    @property
    def random(self):
        return NpRandUtils
