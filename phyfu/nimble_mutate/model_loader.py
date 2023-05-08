from abc import ABC, abstractmethod
import nimblephysics as nimble
import numpy as np
import torch

from phyfu.utils.path_utils import ModulePath
from phyfu.array_utils.array_interface import ArrayUtils
from phyfu.common.model_loader import Model


class NimbleModel(Model, ABC):
    @staticmethod
    @abstractmethod
    def expand_act(action_list):
        ...

    def step(self, init_state, action_list):
        expanded_act = self.expand_act(action_list)
        state = init_state
        for act in expanded_act:
            for i in range(self.substeps):
                state = nimble.timestep(self.world, state, act)
        return state

    def step_trace(self, init_state, action_list, sel_ids):
        expanded_act = self.expand_act(action_list)
        trace = []
        state = init_state
        for i, act in enumerate(expanded_act):
            for _ in range(self.substeps):
                state = nimble.timestep(self.world, state, act)
            if i in sel_ids:
                trace.append(self.state_relevant_part(state))
                # trace.append(self.state_embedding(state))
        return np.array(trace)

    @staticmethod
    @abstractmethod
    def state_relevant_part(state):
       ...

    def two_stage_step(self, init_state: torch.Tensor, action_list: torch.Tensor,
                       stage1_num_steps: int):
        expanded_act = self.expand_act(action_list)
        state = init_state
        stage_1_state = None
        for i, act in enumerate(expanded_act):
            for j in range(self.substeps):
                state: torch.Tensor = nimble.timestep(self.world, state, act)
            if i == stage1_num_steps - 1:
                stage_1_state = state
        return stage_1_state, state

    @property
    def default_state(self):
        return self._default_state

    def mutate_seed_action(self, mut_dev, seed_action, mut_steps):
        return torch.concatenate([seed_action[:mut_steps] * (1 + mut_dev),
                                  seed_action[mut_steps:]])


class TwoBalls(NimbleModel):
    def __init__(self, module_path_utils: ModulePath, au: ArrayUtils,
                 override_options=None):
        super().__init__(module_path_utils, au, override_options)
        self.grace_gap = self._cfg.model_def.grace_gap
        self.substeps = self._cfg.model_def.substeps

        world = nimble.simulation.World()
        world.setGravity([0, 0, 0])
        world.setTimeStep(self._cfg.model_def.dt)

        self.wall_pos = self._cfg.model_def.wall_pos
        radius = self._cfg.model_def.radius
        self.max_vel_allowed = 2. * radius / self._cfg.model_def.dt
        thickness = 0.1
        width = 0.6
        wall_offset = self.wall_pos + radius + thickness / 2
        lf = Ball.create_floor(-wall_offset, "yz", thickness, width)
        rf = Ball.create_floor(wall_offset, "yz", thickness, width)
        world.addSkeleton(lf)
        world.addSkeleton(rf)

        b1 = Ball.create_ball([0, 0, 0], radius, color=[68, 114, 196])
        b2 = Ball.create_ball([0, 0, 0], radius, color=[112, 173, 71])
        world.addSkeleton(b1)
        world.addSkeleton(b2)

        init_pos, init_vel = self._cfg.model_def.init_pos, self._cfg.model_def.init_vel
        world.setPositions(np.expand_dims(np.array([-init_pos, 0., init_pos, 0.]), axis=1))
        world.setVelocities(np.expand_dims(np.array([init_vel, 0., -init_vel, 0.]), axis=1))
        self._default_state = torch.from_numpy(np.copy(world.getState()))

        self.world = world

    def is_valid_state(self, state):
        return torch.max(torch.abs(self.get_velocities(state))).item() < \
            self.max_vel_allowed \
            and \
            torch.max(torch.abs(self.get_positions(state))).item() < \
            abs(self.wall_pos) + abs(self.grace_gap)

    @staticmethod
    def state_relevant_part(state):
        return state[0::2].detach().numpy()

    def state_embedding(self, state):
        return torch.Tensor([state[0].item(), state[2].item(),
                             state[4].item() / self.max_vel_allowed,
                             state[6].item() / self.max_vel_allowed])

    @property
    def name(self):
        return 'two_balls'

    @property
    def act_std(self):
        return torch.tensor(self._cfg.seed_getter.act_std)

    @property
    def dof_shape(self) -> tuple:
        return 2,

    @staticmethod
    def expand_act(action_list):
        return torch.concatenate([
            action_list[:, :1], torch.zeros((len(action_list), 1)),
            action_list[:, 1:2], torch.zeros((len(action_list), 1)),
        ], dim=1)

    def rand_action(self, num_steps):
        return self.rand_act_tile(num_steps)

    @staticmethod
    def get_velocities(state: torch.Tensor):
        return state[4:8:2]

    @staticmethod
    def get_positions(state: torch.Tensor):
        return state[0:4:2]

    def get_mapping(self):
        ikMap: nimble.neural.IKMapping = nimble.neural.IKMapping(self.world)
        ballNode: nimble.dynamics.BodyNode = self.world.getSkeleton(2).getBodyNode(0)
        ikMap.addLinearBodyNode(ballNode)
        return ikMap


class Ball:
    def __init__(self, num_balls=1):
        self.num_balls = num_balls

        world = nimble.simulation.World()
        world.setGravity([0, 0, 0])

        world.setTimeStep(1e-3)

        thickness, width = 0.01, 0.5
        offset = 0.1
        radius = 0.03

        wall_offset = offset + radius + thickness / 2
        right = Ball.create_floor(wall_offset, 'yz', thickness, width)
        left = Ball.create_floor(-wall_offset, 'yz', thickness, width)
        top = Ball.create_floor(wall_offset, 'xz', thickness, width)
        bottom = Ball.create_floor(-wall_offset, 'xz', thickness, width)
        world.addSkeleton(right)
        world.addSkeleton(left)
        world.addSkeleton(top)
        world.addSkeleton(bottom)

        color = [118 * 0.8, 224 * 0.8, 65 * 0.8]
        for i in range(self.num_balls):
            ball = Ball.create_ball([0, offset, 0], radius=radius, color=color)
            world.addSkeleton(ball)

        self.world = world

    @staticmethod
    def create_floor(offset, orientation: str, thickness, width):
        floor = nimble.dynamics.Skeleton()
        floor_joint, floor_body = floor.createWeldJointAndBodyNodePair()
        floor_offset = nimble.math.Isometry3()

        if orientation == 'xy':
            position = [0, 0, offset]
            shape_size = [width, width, thickness]
        elif orientation == 'yz':
            position = [offset, 0, 0]
            shape_size = [thickness, width, width]
        else:
            position = [0, offset, 0]
            shape_size = [width, thickness, width]

        floor_offset.set_translation(position)
        floor_joint.setTransformFromParentBodyNode(floor_offset)

        floor_shape = floor_body.createShapeNode(nimble.dynamics.BoxShape(shape_size))
        floor_visual = floor_shape.createVisualAspect()
        floor_visual.setColor([0.5, 0.5, 0.5])
        floor_visual.setCastShadows(False)
        floor_body.setRestitutionCoeff(1.0)
        floor_shape.createCollisionAspect()
        return floor

    @staticmethod
    def create_ball(origin, radius, color):
        ball = nimble.dynamics.Skeleton()
        sphere_joint, sphere_body = ball.createTranslationalJoint2DAndBodyNodePair()

        offset = nimble.math.Isometry3()
        offset.set_translation(origin)
        sphere_joint.setTransformFromParentBodyNode(offset)

        sphereShape = sphere_body.createShapeNode(nimble.dynamics.SphereShape(radius))
        sphereVisual = sphereShape.createVisualAspect()
        sphereVisual.setColor([i / 255.0 for i in color])
        sphereShape.createCollisionAspect()
        sphere_body.setFrictionCoeff(0.0)
        sphere_body.setRestitutionCoeff(1.0)
        sphere_body.setMass(1)

        return ball

    def get_world_name(self):
        return 'ball'
    
    def get_mapping(self):
        ikMap: nimble.neural.IKMapping = nimble.neural.IKMapping(self.world)
        ballNode: nimble.dynamics.BodyNode = self.world.getSkeleton(4).getBodyNode(0)
        ikMap.addLinearBodyNode(ballNode)
        return ikMap

    def get_mut_vel_dev(self):
        return [1 for _ in range(self.world.getNumDofs())]


class Catapult(NimbleModel):
    def __init__(self, module_path_utils: ModulePath, au: ArrayUtils,
                 override_options=None):
        super().__init__(module_path_utils, au, override_options)
        self.dt = self._cfg.model_def.dt
        self.substeps = self._cfg.model_def.substeps

        world: nimble.simulation.World = nimble.simulation.World()
        # world.setGravity([0, -9.81, 0])
        world.setGravity([0, 0, 0])
        world.setTimeStep(self.dt)

        catapult = nimble.dynamics.Skeleton()

        rootJoint, root = catapult.createWeldJointAndBodyNodePair()
        rootOffset = nimble.math.Isometry3()
        rootOffset.set_translation([0.5, -0.45, 0])
        rootJoint.setTransformFromParentBodyNode(rootOffset)

        def createTailSegment(parent, color):
            poleJoint, pole = catapult.createRevoluteJointAndBodyNodePair(parent)
            poleJoint.setAxis([0, 0, 1])
            poleShape = pole.createShapeNode(nimble.dynamics.BoxShape([.05, 0.25, .05]))
            poleVisual = poleShape.createVisualAspect()
            poleVisual.setColor(color)
            poleJoint.setControlForceUpperLimit(0, 1000.0)
            poleJoint.setControlForceLowerLimit(0, -1000.0)
            poleJoint.setVelocityUpperLimit(0, 10000.0)
            poleJoint.setVelocityLowerLimit(0, -10000.0)

            poleOffset = nimble.math.Isometry3()
            poleOffset.set_translation([0, -0.125, 0])
            poleJoint.setTransformFromChildBodyNode(poleOffset)

            poleJoint.setPosition(0, 90 * 3.1415 / 180)
            poleJoint.setPositionUpperLimit(0, 180 * 3.1415 / 180)
            poleJoint.setPositionLowerLimit(0, 0 * 3.1415 / 180)

            poleShape.createCollisionAspect()
            pole.setFrictionCoeff(0.0)
            pole.setRestitutionCoeff(1.0)

            if parent != root:
                childOffset = nimble.math.Isometry3()
                childOffset.set_translation([0, 0.125, 0])
                poleJoint.setTransformFromParentBodyNode(childOffset)
            return pole

        tail1 = createTailSegment(root, [182.0 / 255, 223.0 / 255, 144.0 / 255])
        tail2 = createTailSegment(tail1, [223.0 / 255, 228.0 / 255, 163.0 / 255])
        tail3 = createTailSegment(tail2, [221.0 / 255, 193.0 / 255, 121.0 / 255])

        catapult.setPositions(np.array([45, 0, 45]) * 3.1415 / 180)
        world.addSkeleton(catapult)

        floor = nimble.dynamics.Skeleton()
        floor.setName('floor')  # important for rendering shadows

        floorJoint, floorBody = floor.createWeldJointAndBodyNodePair()
        floorOffset = nimble.math.Isometry3()
        floorOffset.set_translation([1.2, -0.7, 0])
        floorJoint.setTransformFromParentBodyNode(floorOffset)
        floorShape: nimble.dynamics.ShapeNode = floorBody.createShapeNode(nimble.dynamics.BoxShape(
            [3.5, 0.25, .5]))
        floorVisual: nimble.dynamics.VisualAspect = floorShape.createVisualAspect()
        floorVisual.setColor([0.5, 0.5, 0.5])
        floorVisual.setCastShadows(False)
        floorShape.createCollisionAspect()
        floorBody.setFrictionCoeff(0.0)
        floorBody.setRestitutionCoeff(1.0)

        world.addSkeleton(floor)

        self._default_state = torch.from_numpy(np.copy(world.getState()))
        self._act_std = torch.tensor(self._cfg.seed_getter.act_std)

        self.world = world

    @staticmethod
    def expand_act(action_list):
        return action_list

    def state_embedding(self, state):
        return torch.concatenate(
            [state[:3] / 180., state[3:] / self._cfg.seed_getter.max_vel_allowed])

    @staticmethod
    def state_relevant_part(state):
        return state.detach().numpy()

    @property
    def act_std(self):
        return self._act_std

    @property
    def name(self):
        return "catapult"

    def is_valid_state(self, state):
        return torch.max(torch.abs(state[3:])) < self.config.seed_getter.max_vel_allowed
