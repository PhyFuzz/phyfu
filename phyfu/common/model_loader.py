from abc import ABC, abstractmethod
from omegaconf import DictConfig, OmegaConf
import numpy as np
from typing import Set

from phyfu.utils.path_utils import ModulePath
from phyfu.array_utils.array_interface import ArrayUtils


class Model(ABC):
    def __init__(self, module_path_utils: ModulePath, au: ArrayUtils,
                 override_options: DictConfig = None):
        self._array_utils = au
        self._dof_shape = None
        cfg = OmegaConf.load(module_path_utils.mutate_config_path)
        if override_options is not None:
            cfg = OmegaConf.merge(cfg, override_options)
        self._cfg = cfg

    @property
    @abstractmethod
    def name(self):
        """
        The name of the model
        :return: name, a string
        """

    @property
    def config(self):
        return self._cfg

    def state_embedding(self, state):
        """
        Return an embedding vector to represent the state. The vector will be used to guide
        the process of seed generation.
        :param state: A seed state
        :return: A vector
        """
        return self._array_utils.flatten(state)

    @staticmethod
    def state_relevant_part(state) -> np.ndarray:
        return state

    def get_non_zero_act_steps(self, num_steps):
        max_non_zero_act_steps = min(self.config.max_non_zero_act_steps, num_steps)
        min_non_zero_act_steps = self.config.mut_steps
        non_zero_act_steps = self._array_utils.random.randint(
            min_non_zero_act_steps, max_non_zero_act_steps + 1)
        return non_zero_act_steps

    def rand_act_tile(self, num_steps):
        non_zero_act_steps = self.get_non_zero_act_steps(num_steps)
        act = self._array_utils.random.rand_normal(0, self.act_std, size=self.dof_shape)
        return self._array_utils.concatenate(
            [self._array_utils.tile(act, (non_zero_act_steps, 1)),
             self._array_utils.zeros((num_steps - non_zero_act_steps, *self.dof_shape))])

    def rand_action(self, num_steps):
        """
        Return a series of random action used for a simulation span of num_steps
        :param num_steps: the number of simulation steps
        :return: an action list with length num_steps
        """
        non_zero_act_steps = self.get_non_zero_act_steps(num_steps)
        return self._array_utils.concatenate([
            self._array_utils.random.rand_normal(
                0, self.act_std, size=(non_zero_act_steps, *self.dof_shape)),
            self._array_utils.zeros((num_steps - non_zero_act_steps, *self.dof_shape))
        ])

    @property
    def act_std(self):
        """
        The act_std that will be used as the sigma in rand_normal
        :return: an array or list
        """
        return self.config.seed_getter.act_std

    @property
    def dof_shape(self) -> tuple:
        """
        The shape that will be used to construct mut_dev for mutating seed action
        :return: the shape of mut_dev. Should be a tuple
        """
        if self._dof_shape is None:
            self._dof_shape = \
                np.array(self.config.seed_getter.act_std, dtype=np.float32).shape
        return self._dof_shape

    @abstractmethod
    def step(self, init_state, action_list):
        """
        Run simulation with action list as action_list and starting from init_state
        :param init_state: the initial state for the simulation
        :param action_list: the forces applied on each simulation step
        :return: the state at the end of simulation
        """

    @abstractmethod
    def step_trace(self, init_state, action_list, sel_ids: Set) -> np.ndarray:
        """
        Run forward simulation and collect trace
        :param init_state: the initial state to start the simulation
        :param action_list: the force applied in each step
        :param sel_ids: retrieve states on the steps `sel_ids` of the trace.
        :return: a list of states in numpy
        """

    @property
    @abstractmethod
    def default_state(self):
        """
        The default state of the model
        :return: A state that can be used as the init_state in the step() function
        """

    @abstractmethod
    def two_stage_step(self, init_state, action_list, stage1_num_steps):
        """
        Returns (state after stage1_num_steps) and (state after applying all action_list)
        :param init_state: the initial state for the simulation
        :param action_list: the whole list of actions to be applied during simulation
        :param stage1_num_steps: the step in which an intermediate result
        should be retrieved
        :return: state_stage1, state_final
        """
        ...

    @abstractmethod
    def is_valid_state(self, state):
        """
        Check whether a state, particularly the seed state, is valid or not
        :param state: the state, in particular, the seed state, to be checked
        :return: True or False
        """

    @abstractmethod
    def mutate_seed_action(self, mut_dev, seed_action, mut_steps):
        """
        Mutate the first mut_steps elements of seed_action by mut_dev
        :param mut_dev: the mutant's relative deviation from the seed_action
        :param seed_action: the sequence of action applied on seed state during simulation
        :param mut_steps: the number of mutated elements on the sequence of seed action
        :return: the mutated sequence of action
        """
