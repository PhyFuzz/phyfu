import jax

from phyfu.array_utils.rand_interface import RandUtils


class JaxRandUtils(RandUtils):

    key = jax.random.PRNGKey(0)
    subkey = key

    def __init__(self):
        raise RuntimeError("BraxRandUtil should not be initialized")

    @staticmethod
    def randint(low, high, size=None):
        JaxRandUtils.key, JaxRandUtils.subkey = jax.random.split(JaxRandUtils.key)
        if size is None:
            return jax.random.randint(JaxRandUtils.subkey, (1,), low, high)[0].item()
        else:
            return jax.random.randint(JaxRandUtils.subkey, size, low, high)

    @staticmethod
    def rand_normal(mean, sigma, size):
        JaxRandUtils.key, JaxRandUtils.subkey = jax.random.split(JaxRandUtils.key)
        return jax.random.normal(JaxRandUtils.subkey, size) * sigma + mean
