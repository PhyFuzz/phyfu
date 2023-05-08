import jax
from jax import numpy as jnp

from phyfu.array_utils.array_interface import ArrayUtils
from phyfu.array_utils.jax_random import JaxRandUtils


class JaxArrayUtils(ArrayUtils):
    @staticmethod
    @jax.jit
    def length(v):
        return jnp.linalg.norm(v)

    @staticmethod
    @jax.jit
    def angle(v1, v2):
        u1 = v1 / jnp.linalg.norm(v1)
        u2 = v2 / jnp.linalg.norm(v2)
        return jnp.arccos(jnp.clip(jnp.dot(u1, u2), -1.0, 1.0))

    @staticmethod
    def to_numpy(arr):
        return arr.numpy()

    @staticmethod
    @jax.jit
    def tile(a, rep):
        return jnp.tile(a, rep)

    @staticmethod
    @jax.jit
    def euc_dist(a1, a2):
        return jnp.linalg.norm(a1 - a2)

    @staticmethod
    def zeros(shape):
        return jnp.zeros(shape, dtype=jnp.float32)

    @staticmethod
    def concatenate(arrays):
        return jnp.concatenate(arrays)

    @staticmethod
    def loss_to_float(loss) -> float:
        return loss.item()

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(JaxArrayUtils, cls).__new__(cls)
        return cls.instance

    @property
    def random(self):
        return JaxRandUtils

    @staticmethod
    def l2norm(arr, axis=1, keep_dims=False):
        pass

    @staticmethod
    def max(arr):
        pass

    @staticmethod
    def slice(arr, start=None, end=None):
        pass

    @staticmethod
    def save(file, arr):
        jnp.savez(file, arr)

    @staticmethod
    def load(file):
        return jnp.load(file, allow_pickle=True)['arr_0']