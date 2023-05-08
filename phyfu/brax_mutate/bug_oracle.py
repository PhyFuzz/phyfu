from abc import ABC

from jax import numpy as jnp
import jax
import brax
from brax import pytree

from phyfu.common.bug_oracle import BugOracle


def brax_qp_abs_diff(a, ref):
    return jnp.concatenate([
        jnp.abs(a.pos - ref.pos).flatten(),
        jnp.abs(a.vel - ref.vel).flatten(),
        jnp.abs(a.rot - ref.rot).flatten(),
        jnp.abs(a.ang - ref.ang).flatten()
    ])


def euc_dis(a: jnp.ndarray, b: jnp.ndarray):
    return jnp.linalg.norm(a - b, axis=1)


def cos_dis(a: jnp.ndarray, b: jnp.ndarray):
    return jnp.array([1.0 - jnp.dot(x, y) for x, y in zip(a, b)])


@jax.jit
def euc_diff(a, ref):
    return jnp.concatenate([
        euc_dis(a.pos, ref.pos), euc_dis(a.vel, ref.vel), euc_dis(a.ang, ref.ang),
        jnp.abs(cos_dis(a.rot, ref.rot))
    ])


def brax_qp_rel_diff(a, ref, eps=1e-5):
    return jnp.concatenate([
        jnp.abs(a.pos - ref.pos / (jnp.abs(ref.pos) + eps)).flatten(),
        jnp.abs(a.vel - ref.vel / (jnp.abs(ref.vel) + eps)).flatten(),
        jnp.abs(a.rot - ref.rot / (jnp.abs(ref.rot) + eps)).flatten(),
        jnp.abs(a.ang - ref.ang / (jnp.abs(ref.ang) + eps)).flatten()
    ])


@jax.jit
def diff_clip(diff, threshold):
    return jnp.where(diff > threshold, diff, 0.0)


class BraxOracle(BugOracle, ABC):
    __pytree_ignore__ = ('oracle_cfg',)

    def is_deviated(self, diff_before, diff_after):
        return jnp.any(diff_after > diff_before * self.oracle_cfg.sigma)


@pytree.register
class TwoBallsOracle(BraxOracle):
    # @jax.jit
    # def is_grad_dir_wrong(self, opt_dev: jnp.ndarray, grads: jnp.ndarray):
    #     # return jnp.any(jnp.logical_and(opt_dev * grads < 0, jnp.abs(opt_dev) > 1e-2))
    #     return jnp.all(opt_dev * grads < 0)

    @staticmethod
    def qp_to_metrics(qp):
        return jnp.array([qp.pos[0][0], qp.pos[1][0], qp.vel[0][0], qp.vel[1][0]])

    @jax.jit
    def get_diff(self, mut_init_before, mut_init_after, seed_init):
        mut_init_before = TwoBallsOracle.qp_to_metrics(mut_init_before)
        mut_init_after = TwoBallsOracle.qp_to_metrics(mut_init_after)
        seed_init = TwoBallsOracle.qp_to_metrics(seed_init)
        # Do not clip diff_before since diff_before is in denominator.
        diff_before = jnp.abs(seed_init - mut_init_before)
        diff_after = diff_clip(
            jnp.abs(seed_init - mut_init_after), self.oracle_cfg.diff_tolerance)
        return diff_before, diff_after


@pytree.register
class UR5EOracle(BraxOracle):
    @staticmethod
    def qp_to_dof(qp):
        return brax.QP(
            pos = qp.pos[:-1],
            vel = qp.vel[:-1],
            rot = qp.rot[:-1],
            ang = qp.ang[:-1]
        )

    @jax.jit
    def get_diff(self, mut_init_before, mut_init_after, seed_init):
        mut_init_before = UR5EOracle.qp_to_dof(mut_init_before)
        mut_init_after = UR5EOracle.qp_to_dof(mut_init_after)
        seed_init = UR5EOracle.qp_to_dof(seed_init)
        diff_before = euc_diff(mut_init_before, seed_init)
        diff_after = diff_clip(
            euc_diff(mut_init_after, seed_init), self.oracle_cfg.diff_tolerance)
        return diff_before, diff_after
