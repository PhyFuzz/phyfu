from abc import ABC
from omegaconf import DictConfig
import warp as wp
import warp.sim
from warp.sim import State
import numpy as np
import random
from typing import Union

from phyfu.array_utils.array_interface import ArrayUtils
from phyfu.array_utils.wp_array import NpState
from phyfu.common.model_loader import Model
from phyfu.utils.path_utils import ModulePath
from phyfu.warp_mutate.integrator.euler_two_balls import TwoBallsEuler
from phyfu.warp_mutate.integrator.euler_snake import SnakeEuler


@wp.kernel
def add_mut(seed: wp.array(dtype=wp.float32), mut: wp.array(dtype=wp.float32),
            c: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    c[tid] = seed[tid] * (1. + mut[tid])

def copy_src_to_tgt_state(src: State, tgt: State):
    if src.body_count:
        tgt.body_q.assign(src.body_q)
        tgt.body_qd.assign(src.body_qd)

    if src.particle_count:
        tgt.particle_q.assign(src.particle_q)
        tgt.particle_qd.assign(src.particle_qd)


def clone_state(o: State):
    s = State()

    s.particle_count = o.particle_count
    s.body_count = o.body_count

    s.particle_q = None
    s.particle_qd = None
    s.particle_f = None

    s.body_q = None
    s.body_qd = None
    s.body_f = None

    # particles
    if o.particle_count:
        s.particle_q = wp.clone(o.particle_q)
        s.particle_qd = wp.clone(o.particle_qd)
        s.particle_f = wp.zeros_like(o.particle_qd)

        s.particle_q.requires_grad = o.particle_q.requires_grad
        s.particle_qd.requires_grad = o.particle_qd.requires_grad
        s.particle_f.requires_grad = o.particle_f.requires_grad

    # articulations
    if o.body_count:
        s.body_q = wp.clone(o.body_q)
        s.body_qd = wp.clone(o.body_qd)
        s.body_f = wp.zeros_like(o.body_qd)

        s.body_q.requires_grad = o.body_q.requires_grad
        s.body_qd.requires_grad = o.body_qd.requires_grad
        s.body_f.requires_grad = o.body_f.requires_grad

    return s


class WarpModel(Model, ABC):
    def step(self, init_state: Union[State, NpState], action_list: np.ndarray):
        init_state = self.convert_np_state(init_state)
        action_list = [wp.from_numpy(a, dtype=wp.float32) for a in action_list]
        self.set_init_state(init_state)
        for i, act in enumerate(action_list):
            for j in range(self.substeps):
                step_num = i * self.substeps + j
                self.states[step_num].clear_forces()
                self.integrator.simulate(self.model, act, self.states[step_num],
                                         self.states[step_num+1], self.dt)

        return clone_state(self.states[len(action_list) * self.substeps])

    def two_stage_step(self, init_state: Union[NpState, State],
                       action_list, stage1_num_steps):
        self.step(self.convert_np_state(init_state), action_list)
        return clone_state(self.states[stage1_num_steps * self.substeps]), \
            clone_state(self.states[len(action_list) * self.substeps])

    def step_trace(self, init_state: Union[State, NpState], action_list, sel_ids):
        self.step(self.convert_np_state(init_state), action_list)
        return np.array([self.state_relevant_part(self.states[(i + 1) * self.substeps])
                         for i in sel_ids])

    @staticmethod
    def convert_np_state(state: Union[State, NpState]):
        if isinstance(state, NpState):
            return state.to_wp_state()
        return state

    def set_init_state(self, init_state):
        copy_src_to_tgt_state(init_state, self.states[0])

    def mutate_seed_action(self, mut_dev, seed_action, mut_steps):
        return np.concatenate([
            seed_action[:mut_steps] * (1 + mut_dev),
            seed_action[mut_steps:]
        ])

    @property
    def act_std(self):
        return self._act_std


class TwoBalls(WarpModel):
    def __init__(self, module_path_utils: ModulePath, au: ArrayUtils,
                 override_options: DictConfig = None):
        super().__init__(module_path_utils, au, override_options)

        np.random.seed(self._cfg.seed)
        random.seed(self._cfg.seed)

        self.sim_steps = self._cfg.num_steps
        self.substeps = 1
        self.dt = self._cfg.model_def.dt

        self._act_std = np.array(self._cfg.seed_getter.act_std, dtype=np.float32)
        self._dof_shape = self._act_std.shape

        wall_pos = self._cfg.model_def.wall_pos

        builder = warp.sim.ModelBuilder()

        # default up axis is y
        builder.add_particle(pos=(-self._cfg.model_def.init_pos, 0.0, 0.0),
                             vel=(self._cfg.model_def.init_vel, 0.0, 0.0), mass=1.0)
        builder.add_particle(pos=(self._cfg.model_def.init_pos, 0.0, 0.0),
                             vel=(-self._cfg.model_def.init_vel, 0.0, 0.0), mass=1.0)
        builder.add_shape_box(body=-1, pos=(-wall_pos, 0.0, 0.0),
                              hx=0.01, hy=1.0, hz=1.0)
        builder.add_shape_box(body=-1, pos=(wall_pos, 0.0, 0.0),
                              hx=0.01, hy=1.0, hz=1.0)

        self.model = builder.finalize()

        self.model.wall_p = wp.array([wall_pos, -wall_pos], dtype=wp.float32)
        self.model.wall_n = wp.array([-np.sign(wall_pos), np.sign(wall_pos)],
                                     dtype=wp.float32)

        self.model.particle_radius = float(self._cfg.model_def.radius)
        self.model.customized_wall_x = float(self._cfg.model_def.wall_pos)
        self.model.customized_kn = 1.e4
        self.model.customized_kd = 1.e1

        self.model.ground = False
        self.model.gravity = np.zeros(3, dtype=np.float32)

        self.integrator = TwoBallsEuler()

        max_steps = max(self.sim_steps, self._cfg.seed_getter.max_steps)

        self.states = [self.model.state(requires_grad=True) for _ in range(max_steps + 1)]

        self._default_state = clone_state(self.states[0])

    def rand_action(self, num_steps):
        return self.rand_act_tile(num_steps)

    @property
    def default_state(self):
        return self._default_state

    @property
    def name(self):
        return "two_balls"

    def state_embedding(self, state: Union[NpState, State]):
        if isinstance(state, NpState):
            return np.array([state.q[0][0] / self._cfg.model_def.wall_pos,
                             state.q[1][0] / self._cfg.model_def.wall_pos,
                             state.qd[0][0] / self._cfg.seed_getter.max_vel_allowed,
                             state.qd[1][0] / self._cfg.seed_getter.max_vel_allowed])
        q = state.particle_q.to("cpu").numpy()
        qd = state.particle_qd.to("cpu").numpy()
        return np.array([q[0][0] / self._cfg.model_def.wall_pos,
                         q[1][0] / self._cfg.model_def.wall_pos,
                         qd[0][0] / self._cfg.seed_getter.max_vel_allowed,
                         qd[1][0] / self._cfg.seed_getter.max_vel_allowed])

    @staticmethod
    def state_relevant_part(state: Union[NpState, State]) -> np.ndarray:
        if isinstance(state, NpState):
            return np.array([state.q[0][0], state.q[1][0],
                             state.qd[0][0], state.qd[1][0]], dtype=np.float32)
        else:
            q = state.particle_q.to("cpu").numpy()
            qd = state.particle_qd.to("cpu").numpy()
            return np.array([q[0][0], q[1][0], qd[0][0], qd[1][0]], dtype=np.float32)

    @staticmethod
    def assign_act(s: State, act: np.ndarray):
        s.particle_f.assign([[act[0].item(), 0., 0.], [act[1].item(), 0., 0.]])

    def is_valid_state(self, state: State):
        qd = state.particle_qd.to("cpu").numpy()
        v = qd[:, :3]
        vel_valid = not np.any(
            np.linalg.norm(v, axis=1) > self.config.seed_getter.max_vel_allowed)
        if not vel_valid:
            return False
        q = state.particle_q.to("cpu").numpy()
        wall_limit = abs(self._cfg.model_def.wall_pos)
        pos_valid = np.max([np.abs(q[0][0]), np.abs(q[1][0])]) < wall_limit + \
            self._cfg.model_def.grace_gap
        return pos_valid


class SnakeModel(WarpModel):
    def __init__(self, module_path_utils: ModulePath, au: ArrayUtils,
                 override_options: DictConfig = None):
        super().__init__(module_path_utils, au, override_options)

        np.random.seed(self._cfg.seed)
        random.seed(self._cfg.seed)

        self.substeps = self._cfg.model_def.substeps
        self.dt = 1.0 / (self.substeps * self._cfg.model_def.frame_rate)

        self._act_std = np.array(self._cfg.seed_getter.act_std, dtype=np.float32)
        self._dof_shape = self._act_std.shape

        self.num_chains = self._cfg.model_def.num_chains
        chain_length = self._cfg.model_def.chain_length
        root_height = self._cfg.model_def.root_height
        if self._cfg.model_def.joint_type == "revolute":
            joint_type = wp.sim.JOINT_REVOLUTE
        else:
            joint_type = wp.sim.JOINT_COMPOUND

        builder = wp.sim.ModelBuilder()
        builder.add_articulation()

        for chain_id in range(self.num_chains):
            if chain_id == 0:
                parent = -1
                parent_joint_xform = wp.transform(
                    [0.0, root_height, 0.0], wp.quat_identity())
            else:
                parent = builder.joint_count - 1
                parent_joint_xform = wp.transform(
                    [chain_length, 0.0, 0.0], wp.quat_identity())

            # create body
            b = builder.add_body(
                parent=parent,
                origin=wp.transform([chain_id, root_height, 0.0], wp.quat_identity()),
                joint_xform=parent_joint_xform,
                joint_axis=[0., 0., 1.],
                joint_type=joint_type,
                joint_limit_lower=-np.deg2rad(self._cfg.model_def.joint_limit),
                joint_limit_upper=np.deg2rad(self._cfg.model_def.joint_limit),
                joint_target_ke=0.0,
                joint_target_kd=0.0,
                joint_limit_ke=30.0,
                joint_limit_kd=0.0,
                joint_armature=0.1)

            builder.add_shape_box(
                pos=(chain_length * 0.5, 0.0, 0.0),
                hx=chain_length * 0.5, hy=0.1, hz=0.1,
                density=10.0,
                body=b,
                ke=1e+5,
                kd=0.,
                kf=0.,
                mu=0.
            )

        self.model = builder.finalize()
        self.model.joint_act.requires_grad = True
        self.model.ground = self._cfg.model_def.ground
        if not self._cfg.model_def.gravity:
            self.model.gravity = np.zeros(3, dtype=np.float32)

        self.integrator = SnakeEuler()

        max_steps = max(self._cfg.num_steps, self._cfg.seed_getter.max_steps)
        self.states = [self.model.state(requires_grad=True)
                       for _ in range(max_steps * self.substeps +1)]

        if self._cfg.model_def.collision:
            self.model.collide(self.states[0])

    @property
    def dof_shape(self):
        return self._dof_shape

    @property
    def name(self):
        return "snake"

    def rand_action(self, num_steps):
        return self.rand_act_tile(num_steps)

    @property
    def default_state(self):
        return self.model.state(requires_grad=True)

    def state_embedding(self, state: Union[NpState, State]):
        if isinstance(state, NpState):
            return np.concatenate([
                state.q.flatten() / self._cfg.model_def.joint_limit,
                state.qd[:, :3].flatten() / self.config.seed_getter.max_w_allowed,
                state.qd[:, 3:].flatten() / self.config.seed_getter.max_v_allowed])
        q = state.body_q.to("cpu").numpy()
        qd = state.body_qd.to("cpu").numpy()
        return np.concatenate([
            q.flatten() / self._cfg.model_def.joint_limit,
            qd[:, :3].flatten() / self.config.seed_getter.max_w_allowed,
            qd[:, 3:].flatten() / self.config.seed_getter.max_v_allowed,
        ])

    def is_valid_state(self, state: State):
        qd = state.body_qd.to("cpu").numpy()
        w, v = qd[:, :3], qd[:, 3:]
        vel_valid = not (
                np.any(np.linalg.norm(v, axis=1) > self.config.seed_getter.max_v_allowed)
                or
                np.any(np.linalg.norm(w, axis=1) > self.config.seed_getter.max_w_allowed))
        if not vel_valid or not self._cfg.model_def.ground:
            return vel_valid
        h = state.body_q.to("cpu").numpy()[:, 1]
        return np.all(h > 0.0)
