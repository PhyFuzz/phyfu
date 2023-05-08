import os

import nimblephysics as nimble
import numpy as np
import torch

from phyfu.array_utils.array_interface import ArrayUtils
from phyfu.nimble_mutate.model_loader import NimbleModel
from phyfu.utils.path_utils import ModulePath, ROOT_DIR


class Atlas(NimbleModel):
    def __init__(self, module_path_utils: ModulePath, au: ArrayUtils,
                 override_options=None):
        super().__init__(module_path_utils, au, override_options)

        # Load Skeleton and define necessary properties
        self.dt = self._cfg.model_def.dt
        self.substeps = self._cfg.model_def.substeps

        world = nimble.simulation.World()
        world.setGravity([0, -9.81, 0])

        model_dir = os.path.join(os.path.dirname(ROOT_DIR), "nimblephysics", "data",
                                 "sdf", "atlas")
        atlas: nimble.dynamics.Skeleton = world.loadSkeleton(
            os.path.join(model_dir, "atlas_v3_no_head.urdf"))
        atlas.setPosition(0, -0.5 * 3.14159)
        ground: nimble.dynamics.Skeleton = world.loadSkeleton(
            os.path.join(model_dir, "ground.urdf"))
        floorBody: nimble.dynamics.BodyNode = ground.getBodyNode(0)
        floorBody.getShapeNode(0).getVisualAspect().setCastShadows(False)

        forceLimits = np.ones([atlas.getNumDofs()]) * 500
        forceLimits[0:6] = 0
        atlas.setControlForceUpperLimits(forceLimits)
        atlas.setControlForceLowerLimits(forceLimits * -1)

        self._default_state = torch.from_numpy(np.copy(world.getState()))
        self.world = world

    @property
    def act_std(self):
        # Variation level of the control
        return torch.tensor([self._cfg.act_std for _ in range(27)])

    @property
    def dof_shape(self) -> tuple:
        return 27,

    @staticmethod
    def expand_act(action_list):
        return torch.concatenate([torch.zeros(len(action_list), 6), action_list], dim=1)

    @property
    def name(self):
        return 'atlas'

    def is_valid_state(self, state):
        return True

    def state_embedding(self, state):
        return state.flatten().numpy()

    @staticmethod
    def state_relevant_part(state):
        return state.flatten().numpy()
