from abc import ABC, abstractmethod
from typing import Set

from omegaconf import OmegaConf
import numpy as np
import random
from dataclasses import dataclass
import os

from phyfu.array_utils.array_interface import ArrayUtils
from phyfu.common.model_loader import Model
from phyfu.utils.path_utils import ModulePath


@dataclass
class QP:
    x: np.ndarray
    v: np.ndarray

@dataclass
class MpmAct:
    weights: np.ndarray
    bias: np.ndarray
    num_steps: int


class TaichiModel(Model, ABC):
    def mutate_seed_action(self, mut_dev, seed_action, mut_steps):
        return np.concatenate([seed_action[:mut_steps] * (1 + mut_dev),
                               seed_action[mut_steps:]])

    @staticmethod
    def np_to_taichi(taichi_array, np_array: np.ndarray, name: str):
        if hasattr(taichi_array, 'n'):
            exp_shape = (*taichi_array.shape, taichi_array.n)
        else:
            exp_shape = taichi_array.shape
        if np_array.shape != exp_shape:
            raise RuntimeError(f"Mismatched {name} shape: expected: {exp_shape}, "
                               f"actual shape: {np_array.shape}")
        taichi_array.from_numpy(np_array)

    @staticmethod
    def padded_np_to_taichi(taichi_array, np_array: np.ndarray, name: str):
        if len(np_array) < taichi_array.shape[0]:
            np_array = np.concatenate([
                np_array, np.zeros((taichi_array.shape[0] - len(np_array),
                                    *np_array.shape[1:]), dtype=np.float32)])
        TaichiModel.np_to_taichi(taichi_array, np_array, name)

    def step_trace(self, init_state, action_list, sel_ids: Set) -> np.ndarray:
        self.step(init_state, action_list)
        trace = []
        for s in sel_ids:
            state = self.retrieve_state(s + 1)
            trace.append(np.hstack([state.x.flatten(), state.v.flatten()]))
        return np.array(trace)

    @abstractmethod
    def retrieve_state(self, step_num) -> QP:
        ...

    @abstractmethod
    def loss_and_grads(self, mut_dev: np.ndarray):
        ...

    @abstractmethod
    def before_opt(self, root_state: QP, seed_final: QP, seed_ctrl: np.ndarray):
        ...


class MpmModel(TaichiModel):
    def __init__(self, module_path_utils: ModulePath, au: ArrayUtils,
                 override_options=None):
        super().__init__(module_path_utils, au, override_options)
        np.random.seed(self._cfg.seed)
        random.seed(self._cfg.seed)
        self.substeps = self._cfg.model_def.substeps
        try:
            self._act_std = self._cfg.seed_getter.act_std[0]
        except TypeError:
            self._act_std = self._cfg.seed_getter.act_std
        # The mpm package should only be imported in a single place,
        # otherwise there would be some wierd issues.
        from phyfu.taichi_mutate import mpm
        self.mpm = mpm
        self.mpm.init_scene()
        arr_dir = os.path.join(os.path.dirname(__file__), "examples")
        self.default_weights = np.load(os.path.join(arr_dir, "weights.npy"))\
            .astype(np.float32)
        self.default_bias = np.load(os.path.join(arr_dir, "bias.npy")).astype(np.float32)

    @property
    def name(self):
        return "mpm"

    @property
    def config(self):
        return self._cfg

    @property
    def default_state(self):
        return QP(
            x=self.mpm.default_x,
            v=np.zeros((self.mpm.n_particles, self.mpm.v.n), dtype=np.float32),
        )

    def is_valid_state(self, state: QP):
        return self.mpm.is_state_valid(state.x.flatten())

    def rand_action(self, num_steps):
        return MpmAct(
            weights=self.default_weights *
                    (1. + np.random.randn(4, 4).astype(np.float32) * self._act_std),
            bias=self.default_bias *
                 (1. + np.random.randn(4).astype(np.float32) * self._act_std),
            num_steps=num_steps
        )

    def mutate_seed_action(self, mut_dev, seed_action, mut_steps):
        return MpmAct(
            weights=seed_action.weights * (1. + mut_dev[:16].reshape(4, 4)),
            bias=seed_action.bias * (1. + mut_dev[16:]),
            num_steps=seed_action.num_steps
        )

    @property
    def dof_shape(self) -> tuple:
        return 20,

    def retrieve_state(self, step_num):
        return QP(
            x=np.array([self.mpm.x[step_num * self.substeps, i].to_numpy()
                        for i in range(self.mpm.n_particles)]),
            v=np.array([self.mpm.v[step_num * self.substeps, i].to_numpy()
                        for i in range(self.mpm.n_particles)])
        )

    def state_embedding(self, state: QP):
        return np.vstack([state.x[::600], state.v[::600]]).flatten()

    def set_init_state(self, s: QP):
        self.mpm.copy_init_state(s.x, s.v)

    def step(self, init_state: QP, action_list: MpmAct):
        # memory_layout of action_list: (weights (16), bias (4))
        TaichiModel.np_to_taichi(
            self.mpm.weights, action_list.weights, "weights")
        TaichiModel.np_to_taichi(self.mpm.bias, action_list.bias, "bias")
        self.set_init_state(init_state)
        num_steps = action_list.num_steps
        self.mpm.reset_n_invalid()
        for s in range(num_steps * self.substeps):
            self.mpm.advance(s)
            if self.mpm.encountered_invalid():
                raise RuntimeError(f"Encountered invalid state in mpm step {s}")
        return self.retrieve_state(num_steps)

    def before_opt(self, root_state: QP, seed_final: QP, seed_ctrl: MpmAct):
        TaichiModel.padded_np_to_taichi(
            self.mpm.seed_weights, seed_ctrl.weights, "seed_weights")
        TaichiModel.padded_np_to_taichi(self.mpm.seed_bias, seed_ctrl.bias, "seed_bias")
        TaichiModel.np_to_taichi(self.mpm.ref_x, seed_final.x, "seed_final.x")
        TaichiModel.np_to_taichi(self.mpm.ref_v, seed_final.v, "seed_final.v")
        self.set_init_state(root_state)

    def loss_and_grads(self, mut_dev: np.ndarray):
        TaichiModel.np_to_taichi(self.mpm.mut_weights, mut_dev[:16].reshape(4, 4),
                                 "mut_weights")
        TaichiModel.np_to_taichi(self.mpm.mut_bias, mut_dev[16:], "mut_bias")
        return self.mpm.loss_and_grads(self._cfg.num_steps * self.substeps,
                                       self._cfg.mut_steps * self.substeps)

    def two_stage_step(self, init_state, action_list, stage1_num_steps):
        final_state = self.step(init_state, action_list)
        return self.retrieve_state(stage1_num_steps), final_state


class TwoBallsModel(TaichiModel):
    def __init__(self, module_path_utils: ModulePath, au: ArrayUtils,
                 override_options=None):
        super().__init__(module_path_utils, au, override_options)
        OmegaConf.save(self._cfg, module_path_utils.mutate_config_path)
        from phyfu.taichi_mutate import two_balls
        self.m = two_balls

    def rand_action(self, num_steps):
        # return np.tile(self._array_utils.random.rand_normal(
        #     0, np.array(self.config.seed_getter.act_std),
        #     (self.m.num_objs, self.m.state_size)
        # ), reps=(num_steps, 1, 1))
        return self._array_utils.random.rand_normal(
            0, np.array(self.config.seed_getter.act_std),
            (num_steps, *self.dof_shape)
        )

    def state_embedding(self, state: QP):
        return np.concatenate(
            [state.x.flatten(),
             state.v.flatten() / self.config.seed_getter.max_vel_allowed])

    @property
    def dof_shape(self) -> tuple:
        return self.m.num_objs, self.m.state_size

    @property
    def name(self):
        return "two_balls"

    @property
    def config(self):
        return self.m.cfg

    @property
    def default_state(self):
        return QP(
            x=np.array(self.m.world.init_pos, dtype=np.float32),
            v=np.array(self.m.world.init_vel, dtype=np.float32)
        )

    def retrieve_state(self, step_num):
        x_state, v_state = [], []
        for i in range(self.m.num_objs):
            x_state.append(self.m.x[step_num, i].to_numpy())
            v_state.append(self.m.v[step_num, i].to_numpy())
        x_state, v_state = np.array(x_state), np.array(v_state)
        return QP(x_state, v_state)

    def initialize_before_forward(self, init_state: QP, ctrl_list: np.ndarray):
        self.m.reset_before_forward()
        self.set_init_state(init_state)
        self.set_ctrls(ctrl_list)

    def set_ctrls(self, ctrl_list: np.ndarray):
        TaichiModel.padded_np_to_taichi(self.m.ctrls, ctrl_list, "ctrls")

    def set_init_state(self, init_state: QP):
        if init_state.x.shape != (self.m.x.shape[1], self.m.x.n,) \
                or init_state.v.shape != (self.m.v.shape[1], self.m.v.n):
            raise RuntimeError(
    f"init_state shape mismatch! Required shape: "
    f"{(self.m.x.shape[1], self.m.x.n), (self.m.v.shape[1], self.m.v.n)}, "
    f"input init_state shape: {(init_state.x.shape, init_state.v.shape)}")

        self.m.init_states(init_state.x, init_state.v)

    def step(self, init_state: QP, ctrl_list: np.ndarray):
        self.initialize_before_forward(init_state, ctrl_list)
        num_steps = len(ctrl_list)
        self.m.forward(num_steps)
        return self.retrieve_state(num_steps)

    @staticmethod
    def state_relevant_part(state):
        return state

    def two_stage_step(self, init_state, action_list, stage1_num_steps):
        total_num_steps = len(action_list)
        self.step(init_state, action_list)
        return self.retrieve_state(stage1_num_steps), self.retrieve_state(total_num_steps)

    def is_valid_state(self, state: QP):
        if np.any([np.linalg.norm(v) > self.config.seed_getter.max_vel_allowed
                   for v in state.v]):
            return False
        for i in range(self.m.num_objs):
            # Caution: The num_planes is hard-coded to be 2.
            if np.any(state.x[i] < self.m.world.plane_pos[0]) or \
                np.any(state.x[i] > self.m.world.plane_pos[1]):
                return False
        return True

    def before_opt(self, root_state: QP, seed_final: QP, seed_ctrl: np.ndarray):
        self.set_init_state(root_state)
        self.set_ctrls(seed_ctrl)
        self.np_to_taichi(self.m.seed_to_mut, seed_ctrl[:self.m.mut_steps],
                                   "seed_to_mut")
        TaichiModel.np_to_taichi(self.m.ref_x, seed_final.x, "ref_x")
        TaichiModel.np_to_taichi(self.m.ref_v, seed_final.v, "ref_v")

    def loss_and_grads(self, mut_dev: np.ndarray):
        self.m.reset_before_forward()
        TaichiModel.np_to_taichi(self.m.mut_dev, mut_dev, "mut_dev")
        # return self.m.loss[None], self.m.forward_and_get_grad(self.config.num_steps)
        return self.m.forward_and_get_grad(self.config.num_steps)
