from abc import abstractmethod, ABC
from typing import Any, Tuple

from phyfu.utils.loss_utils import LossUtils
from phyfu.common.model_loader import Model
from phyfu.common.loss_func import LossFunction
from phyfu.utils.log_utils import FuzzingLogger, OptInfo
from phyfu.array_utils.array_interface import ArrayUtils


class Optimizer(ABC):
    def __init__(self, model: Model, loss_func: LossFunction, array_utils: ArrayUtils):
        self.loss_cfg = model.config.loss_utils
        self.model = model
        self.loss_func = loss_func
        self.array_utils = array_utils
        self.loss_utils = None
        self.optimizer = None
        self.optimizer_state = None

    def reset_loss_utils(self):
        self.loss_utils = LossUtils(
            loss_threshold=self.loss_cfg['converge_threshold'],
            max_epochs=self.loss_cfg['max_epochs'],
            max_len=self.loss_cfg['max_len'],
            display_freq=self.loss_cfg['display_freq'])

    @abstractmethod
    def before_optimization(self, mut_dev, root_state, seed_final, seed_act):
        """
        Prepare before the optimization loops start, e.g., setting up an optimizer and
        initialize optimizer_state
        :return: None
        """

    @abstractmethod
    def update_params(self, opt_dev, grads):
        """
        Update opt_dev and optimizer state using gradient descend with the computed gradients
        :param opt_dev: the parameter to update upon
        :param grads: previously computed gradients on opt_dev
        :return: the updated opt_dev
        """

    @abstractmethod
    def compute_loss_and_grads(self, opt_dev, root_state,
                               seed_final, seed_act) -> Tuple[float, Any]:
        """
        Compute the loss and gradients on the given opt_Dev
        :param opt_dev: the deviation to optimize on
        :param root_state: starting state of simulation
        :param seed_final: final state of simulation with seed action
        :param seed_act: the action applied during the seed simulation
        :return: loss, grads
        """

    def gradient_descend(self, mut_dev, mut_final, root_state,
                         seed_final, seed_act, logger: FuzzingLogger):
        opt_dev = mut_dev
        opt_iter = 0
        if logger is not None:
            init_loss = self.array_utils.loss_to_float(
                self.loss_func.apply(mut_final, seed_final))
            opt_info = OptInfo(opt_iter, opt_dev, init_loss, opt_dev)
            logger.log_opt_info(opt_info)
        self.reset_loss_utils()
        self.before_optimization(mut_dev, root_state, seed_final, seed_act)
        while not self.loss_utils.has_converged():
            opt_iter += 1
            loss_value, grads = self.compute_loss_and_grads(
                opt_dev, root_state, seed_final, seed_act)
            self.loss_utils.add_item(self.array_utils.loss_to_float(loss_value), opt_dev)
            if logger is not None:
                opt_info = OptInfo(
                    opt_iter, opt_dev, self.array_utils.loss_to_float(loss_value), grads)
                logger.log_opt_info(opt_info)
            opt_dev = self.update_params(opt_dev, grads)

        print(self.loss_utils.terminate_message)
        return self.loss_utils.get_min_loss(), self.loss_utils.get_min_loss_item(), \
            self.loss_utils.terminate_message
