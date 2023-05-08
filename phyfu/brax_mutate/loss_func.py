from phyfu.common.loss_func import LossFunction

import jax
import brax


class BraxLossFunction(LossFunction):
    @staticmethod
    @jax.jit
    def square_loss(x: brax.QP, y: brax.QP):
        return jax.lax.square(x.pos - y.pos).sum() + \
            jax.lax.square(x.vel - y.vel).sum() + \
            jax.lax.square(x.rot - y.rot).sum() + \
            jax.lax.square(x.ang - y.ang).sum()

    @staticmethod
    @jax.jit
    def linear_loss(x: brax.QP, y: brax.QP):
        return jax.lax.abs(x.pos - y.pos).sum() + \
            jax.lax.abs(x.vel - y.vel).sum() + \
            jax.lax.abs(x.rot - y.rot).sum() + \
            jax.lax.abs(x.ang - y.ang).sum()


