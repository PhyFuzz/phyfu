.. highlight:: shell

============
Extension
============

We will walk through the process of extending PhyFu to support the fuzzing of a new physical scenario. We will use the example of an ``Atlas`` scene. The ``Atlas`` scene is a human-like robot standing on the ground. The robot is able to perform complex actions like doing Yoga. The robot is controlled by dozens of controllers on its joints. The ``Atlas`` scene is a good example of a complex physical scenario.

First, make sure you are at the top directory of the ``phyfu`` repository (the directory with ``setup.py`` inside). Then download the necessary skeleton files for the ``Atlas`` body and ground. For simplicity, we directly clone the original repository of nimblephysics, which contains the skeleton files for the ``Atlas`` scene. The skeleton files are located at ``nimblephysics/data/sdf/atlas``. To clone the repository, run the following command:

.. code-block:: console

    git clone https://github.com/keenon/nimblephysics.git

Next, create a new file ``atlas.py`` inside the directory ``phyfu/nimble_mutate``. Import the necessary modules:

.. code-block:: python3

    import os

    import nimblephysics as nimble
    import numpy as np
    import torch

    from phyfu.array_utils.array_interface import ArrayUtils
    from phyfu.nimble_mutate.model_loader import NimbleModel
    from phyfu.utils.path_utils import ModulePath

Next, create a new class ``Atlas`` that inherits from ``NimbleModel``:

.. code-block:: python3

    class Atlas(NimbleModel):
        def __init__(self, module_path_utils: ModulePath, au: ArrayUtils,
                    override_options=None):
            super().__init__(module_path_utils, au, override_options)

In the ``__init__`` function, we need to load the skeletons that we just downloaded, and define necessary properties, such as the gravity and simulation time step, for the simulation:

.. code-block:: python3

    class Atlas(NimbleModel):
        def __init__(self, module_path_utils: ModulePath, au: ArrayUtils,
                    override_options=None):
            super().__init__(module_path_utils, au, override_options)

            # Load Skeleton and define necessary properties
            self.dt = self._cfg.model_def.dt
            self.substeps = self._cfg.model_def.substeps

            world = nimble.simulation.World()
            world.setGravity([0, -9.81, 0])

            atlas: nimble.dynamics.Skeleton = world.loadSkeleton("atlas_v3_no_head.urdf")
            atlas.setPosition(0, -0.5 * 3.14159)
            ground: nimble.dynamics.Skeleton = world.loadSkeleton("ground.urdf")
            floorBody: nimble.dynamics.BodyNode = ground.getBodyNode(0)
            floorBody.getShapeNode(0).getVisualAspect().setCastShadows(False)

            forceLimits = np.ones([atlas.getNumDofs()]) * 500
            forceLimits[0:6] = 0
            atlas.setControlForceUpperLimits(forceLimits)
            atlas.setControlForceLowerLimits(forceLimits * -1)

            self._default_state = torch.from_numpy(np.copy(world.getState()))
            self.world = world

Note that variable ``self._cfg`` actually stores customizable parameters for the simulation and is loaded from a dedicated ``yaml`` file for easier management of all the parameters. We will see how to customize these parameters later. But first, let's move on to implement other necessary functions. We need to implement functions related to ``action`` so that the robot can move:

.. code-block:: python3

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

In the code above, we define that the robot has 27 degrees of freedom, and the first 6 degrees of freedom are fixed. We also define the variation level of the control (``act_std``) and the function ``expand_act`` to expand the action list to the correct shape.

Besides, we need to implement some methods for the seed scheduling algorithm to work. The seed scheduling algorithm requires a numpy vector as input, so we need to implement a function to convert the state of the robot to a numpy vector:

.. code-block:: python3

    def state_embedding(self, state):
        return state.flatten().numpy()
    
    @staticmethod
    def state_relevant_part(state):
        return state.flatten().numpy()

We also need some additional functions:

.. code-block:: python3

    @property
    def name(self):
        return 'atlas'

    def is_valid_state(self, state):
        return True

We set the ``is_valid_state`` to always return ``True`` because we do not have any constraints on the state of the robot (the ``is_valid_state`` only function as a safeguard if the PSE outputs an invalid state (true positive in this case) even if the simulation is started from a valid meta state.).


Now, we come to the parameter configuration file, which will be loaded and used to populate the ``self._cfg`` variable:


.. code-block:: yaml

    model_def:
        name: atlas
        dt: 1e-3
        substeps: 1
    disable_logging: False
    num_steps: 100
    test_times: 100
    lr: 5e-3
    loss_func: linear
    opt: Adam
    seed_getter:
        type: art
        art_params:
            init_pop_size: 10
            cand_size: 10
            refresh_prob: 0.1
        min_steps: 50
        max_steps: 300
        reset_freq: 20
    act_std: 1
    max_non_zero_act_steps: 20
    mut_dev: 0.05
    mut_steps: 10
    loss_utils:
        max_epochs: 500
        threshold_sigma: 3
        max_len: 100
        display_freq: 10
        converge_threshold: 1.0e-3
    use_gui: False


The content above should be stored in the ``yaml`` file of ``phyfu/configs/fuzzing/nimble/atlas/mutate.yaml``. Notably, the ``model_def.dt`` and ``model_def.substeps`` define the simulation time step and the number of substeps per simulation step, respectively. The ``seed_getter`` defines the parameters related to our Simulate-Then-Collect (STC) scheme. The ``min_steps`` and ``max_steps`` define the minimum and maximum number of simulation steps between the previous collected seed state and the next collected one. The ``reset_freq`` defines the frequency of resetting the STC process. The ``max_non_zero_act_steps`` defines the maximum number of simulation steps that the robot can perform without any action. The ``mut_dev`` and ``mut_steps`` define the standard deviation and the number of steps for the mutation algorithm. The ``loss_utils`` defines the parameters for the loss function. The ``use_gui`` defines whether to use the GUI for visualization (actually this option is not used in the current version of the code).

Also, we need to add the file ``analysis.yaml`` into the same directory as the ``mutate.yaml`` file. This file defines the parameters for oracle checking. The content of the file is as follows:

.. code-block:: yaml

    min_loss_threshold: 1e-1
    diff_tolerance: 1e-3
    sigma: 3
    write_to_file: True

The file contains the parameters for the oracle checking algorithm. The ``min_loss_threshold`` defines the minimum loss value; if the loss value after optimization is greater than the threshold, then we deem it as a backward error. ``sigma`` defines maximum threshold for the forward oracle. Denote difference between the seed and mutant's initial state *before the optimization* as d0, and the difference between the seed and mutant's initial state *after the optimization* as d1. If d1 > d0 * sigma, then we deem it as a forward error. In reality, ``sigma`` is usually set to 3 (the intuition comes from three-sigma rule). Also, to filter numerical noise, we set the elements in d1 that are less than ``diff_tolerance`` to 0.
The ``write_to_file`` defines whether to write the bug information to a file.

The full code for the ``atlas.py`` file is provided in the ``phyfu/nimble_mutate`` folder; the ``yaml`` configuration files are in ``phyfu/configs/fuzzing/nimble/atlas/mutate.yaml`` and ``phyfu/configs/fuzzing/nimble/atlas/analysis.yaml``.

Before we can run the simulation of ``Atlas``, we need to register it so that we can use it from the command line. To do so, we first need to append an entry in the ``factory`` dict of the ``phyfu.nimble_mutate.registry.NimbleRegistry`` class:

.. code-block:: python3

    factory = {
        "two_balls": {
            "fuzz": model_loader.TwoBalls,
            "find_errors": bug_oracle.NimbleOracle
        },
        "catapult": {
            "fuzz": model_loader.Catapult,
            "find_errors": bug_oracle.NimbleOracle
        },
        "atlas": {
            "fuzz": Atlas,
            "find_errors": bug_oracle.NimbleOracle
        }
    }

We also need to add an option to the command line parser in ``phyfu/utils/cli_utils.py``:

.. code-block:: python3

    MODEL_NAME_CHOICES = ["two_balls", "ur5e", "catapult", "snake", "mpm", "atlas"]

Now, we can run the simulation with the following command:

.. code-block:: console

    phyfu.fuzz nimble atlas --test_times 10

After waiting for around 2 minutes, the simulation will finish and output the following information:

.. code-block:: text

    #loss_too_large: 5
    #deviated_init_state: 1

The full code after all the extension steps are provided a separate branch in the repository. The branch can be checked out with the following command:

.. code-block:: console

    git checkout atlas

And you can directly run the fuzzing campaign with the same command as above.
