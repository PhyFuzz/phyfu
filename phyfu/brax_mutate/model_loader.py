import brax
from brax import jumpy as jp
import jax
from brax import pytree
from brax import QP
import jax.numpy as jnp
import numpy as np

from google.protobuf import text_format
from functools import partial

from phyfu.array_utils.array_interface import ArrayUtils
from phyfu.common.model_loader import Model

from phyfu.array_utils.jax_random import JaxRandUtils
from phyfu.brax_mutate import model_def
from phyfu.utils.path_utils import ModulePath


def add_all_pairs(pb):
    bodies = pb.bodies
    for idx1, b1 in enumerate(bodies):
        for idx2 in range(idx1 + 1, len(bodies)):
            b2 = bodies[idx2]
            c = pb.collide_include.add()
            c.first = b1.name
            c.second = b2.name


def add_floor_only(pb):
    for b in pb.bodies:
        if b.name == 'floor':
            continue
        c = pb.collide_include.add()
        c.first = "floor"
        c.second = b.name


def vec_len(arr):
    return jnp.linalg.norm(arr, 2, axis=1)

def exceeds_max(aspect, max_allowed):
    return jnp.max(vec_len(aspect)) > max_allowed

# def exceeds_max_vel_len(qp, max_v_allowed, max_w_allowed):
#     return jnp.max(vec_len(qp.vel)) > max_v_allowed \
#         or jnp.max(vec_len(qp.ang)) > max_w_allowed


@pytree.register
class BraxModel(Model):
    __pytree_ignore__ = ('_cfg', '_default_qp', '_array_utils', '_dof_shape')

    @property
    def default_state(self):
        return self._default_qp

    # @jax.jit
    # def step_fixed(self, init_qp):
    #     def do_one_step(state, _):
    #         next_state, _ = self._world.step(state, jnp.zeros(self.action_size))
    #         return next_state, None
    #
    #     return jax.lax.scan(do_one_step, init_qp, None,
    #                         length=self._cfg['num_steps'])[0]

    @jax.jit
    def step(self, init_qp, action_list):
        def do_one_step(state, action):
            next_state, _ = self._world.step(state, action)
            return next_state, None

        return jax.lax.scan(do_one_step, init_qp, action_list)[0]

    def two_stage_step(self, init_state, action_list, stage1_num_steps):
        stage1_final = self.step(init_state, action_list[:stage1_num_steps])
        stage2_final = self.step(init_state, action_list)
        return stage1_final, stage2_final

    def step_trace(self, init_state, action_list, sel_ids) -> np.ndarray:
        qp, qp_history = self.step_his(init_state, action_list)
        return np.array([[p[0][0], p[1][0], v[0][0], v[1][0]]
                         for i, (p, v) in enumerate(zip(qp_history.pos, qp_history.vel))
                         if i in sel_ids], dtype=np.float32)

    @jax.jit
    def step_his(self, init_qp, action_list):
        def do_one_step(state, action):
            next_state, _ = self._world.step(state, action)
            return next_state, next_state

        qp, qp_history = jax.lax.scan(do_one_step, init_qp, action_list)
        return qp, qp_history

    def rand_action(self, num_steps):
        return jp.concatenate([
            JaxRandUtils.rand_normal(0, s, size=(num_steps, 1))
            for s in self.config.seed_getter.act_std
        ], axis=1)

    @property
    def world(self):
        return self._world

    @property
    def name(self):
        return self.config.model_def.name

    def reset(self):
        return self._default_qp


@pytree.register
class TwoBalls(BraxModel):
    def __init__(self, module_path_utils: ModulePath, au: ArrayUtils,
                 override_options=None):
        super().__init__(module_path_utils, au, override_options)
        world_config = text_format.Parse(model_def.TWO_BALLS_DEF, brax.Config())
        add_all_pairs(world_config)
        self._world = brax.System(world_config)
        self._default_qp = self._world.default_qp()

    @jax.jit
    def state_embedding(self, qp: QP):
        return jnp.array([qp.pos[0][0], qp.pos[1][0],
                          qp.vel[0][0] / self.config.seed_getter.max_vel_allowed,
                          qp.vel[1][0] / self.config.seed_getter.max_vel_allowed])

    @staticmethod
    def state_relevant_part(state: QP) -> np.ndarray:
        return np.array([state.pos[0][0].item(), state.pos[1][0].item(),
                         state.vel[0][0].item(), state.vel[1][0].item()], dtype=np.float32)

    @staticmethod
    def dof_indices():
        return 0, 3

    def rand_action(self, num_steps):
        max_non_zero_act_steps = min(self.config.max_non_zero_act_steps, num_steps)
        min_non_zero_act_steps = self.config.mut_steps
        non_zero_act_steps = JaxRandUtils.randint(
            min_non_zero_act_steps, max_non_zero_act_steps)
        # non_zero_act_indices = list(
        #     set(BraxRandUtil.randint(0, num_steps, size=(non_zero_act_steps,)).tolist()))
        # non_zero_act_indices.sort()
        # non_zero_act_steps = len(non_zero_act_indices)
        non_zero_act = super().rand_action(non_zero_act_steps)
        # print(non_zero_act_indices)
        # print(non_zero_act)
        return jnp.zeros((num_steps, 6), dtype=jnp.float32) \
                   .at[:non_zero_act_steps, TwoBalls.dof_indices()] \
            .set(non_zero_act, unique_indices=True, indices_are_sorted=True)
        # non_zero_act = super().rand_action(num_steps)
        # return TwoBalls.scatter(non_zero_act, num_steps)
        # The 2 means the action size is 2
        # return jnp.array([[i, 0, 0, j, 0, 0] for i, j in non_zero_act])
        # return jnp.concatenate([non_zero_act, jnp.zeros(
        #     (num_steps - self.config.mut_steps, 2), dtype=jnp.float32)])
        # act_list = jnp.zeros((num_steps, 2), dtype=jnp.float32)
        # return act_list.at[self.dof_indices()].set(act)

    @staticmethod
    @partial(jax.jit, static_argnames=['mut_steps'])
    def mutate_seed_action(mut_dev, seed_action, mut_steps):
        # return seed_action
        # return jnp.concatenate(
        #     [jnp.array([[i[0] * (1 + mut_dev[0]), 0, 0, i[3] * (1 + mut_dev[1]), 0, 0]
        #       for i in seed_action[:mut_steps]]),
        #      seed_action[mut_steps:]])
        return seed_action.at[:mut_steps, TwoBalls.dof_indices()]\
            .multiply(1 + mut_dev, unique_indices=True, indices_are_sorted=True)
        # mutated_part = seed_action[:mut_steps] * (1 + mut_dev)
        # return jnp.concatenate([mutated_part, seed_action[mut_steps:]])
        # act_list = jnp.zeros_like(seed_action)
        # return act_list.at[TwoBalls.dof_indices()].set(
        #     seed_action[TwoBalls.dof_indices()] * (1 + mut_dev)
        # )

    def is_valid_state(self, state):
        # The origin of planes is located at -1.0 and 1.0, respectively,
        # so do not compare with state.pos[2][0] and state.pos[3][0]
        return not exceeds_max(state.vel, self.config.seed_getter.max_vel_allowed) \
            and -1.0 < state.pos[0][0] < 1.0 and -1.0 < state.pos[1][0] < 1.0


@pytree.register
class UR5E(BraxModel):
    def __init__(self, module_path_utils: ModulePath, au: ArrayUtils,
                 override_options=None):
        super().__init__(module_path_utils, au, override_options)
        pb = text_format.Parse(model_def.UR5E_DEF, brax.Config())
        add_floor_only(pb)
        self._world = brax.System(pb)
        self._default_qp = self._world.default_qp()

    @jax.jit
    def state_embedding(self, qp: QP):
        return jnp.concatenate([
            qp.pos[:-1].flatten(),
            qp.vel[:-1].flatten() / self.config.seed_getter.max_v_allowed,
            qp.ang[:-1].flatten() / self.config.seed_getter.max_w_allowed,
            qp.rot[:-1].flatten()])

    def step_trace(self, init_state, action_list, sel_ids) -> np.ndarray:
        qp, qp_history = self.step_his(init_state, action_list)
        return np.array([np.hstack([p[:-1].flatten(), r[:-1].flatten(),
                          v[:-1].flatten(), a[:-1].flatten()
                          ])
                         for i, (p, r, v, a) in
                         enumerate(zip(qp_history.pos, qp_history.rot,
                                       qp_history.vel, qp_history.ang))
                         if i in sel_ids], dtype=np.float32)

    @staticmethod
    @partial(jax.jit, static_argnames=['mut_steps'])
    def mutate_seed_action(mut_dev, seed_action, mut_steps):
        return seed_action.at[:mut_steps].multiply(1 + mut_dev)

    def is_valid_state(self, state):
        return not exceeds_max(state.vel, self.config.seed_getter.max_v_allowed) \
            and not exceeds_max(state.ang, self.config.seed_getter.max_w_allowed)
