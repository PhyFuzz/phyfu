from phyfu.common.model_loader import Model
from phyfu.array_utils.array_interface import ArrayUtils


class Mutator:
    def __init__(self, model: Model, au: ArrayUtils):
        self.model = model
        self.cfg = self.model.config
        self.au = au
        self.dof_shape = self.model.dof_shape

    def mutate_action(self, seed_action):
        """
        Mutate the seed_action and return the relative diff mut_dev and the action
        after mutation mut_act
        :param seed_action: the action to be mutated upon
        :return: mut_dev, mut_act
        """
        mut_dev = self.au.random.rand_normal(
            0, self.cfg.mut_dev, size=self.dof_shape)
        mut_act = self.model.mutate_seed_action(mut_dev, seed_action, self.cfg.mut_steps)
        return mut_dev, mut_act

    def mutate(self, root_state, seed_action):
        """
        Mutate the seed_action and return the results by applying mutated action on root_state
        :param root_state: the initial state to start the simulation
        :param seed_action: the action to be mutated upon
        :return: mut_init, mut_final, mut_act, mut_dev
        """
        mut_dev, mut_act = self.mutate_action(seed_action)
        mut_init, mut_final = self.model.two_stage_step(
            root_state, mut_act, self.cfg.mut_steps)
        return mut_init, mut_final, mut_act, mut_dev
