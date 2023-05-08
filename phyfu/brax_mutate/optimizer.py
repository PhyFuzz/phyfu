import jax
import optax
from typing import Any

from phyfu.common.optimizer import Optimizer
from phyfu.common.loss_func import LossFunction
from phyfu.common.model_loader import Model


class BraxOptimizer(Optimizer):
    model: Model
    loss_func: LossFunction
    optimizer: Any

    def __init__(self, model: Model, loss_func: LossFunction, array_utils):
        super().__init__(model, loss_func, array_utils)
        # The act_opt requires the following three fields to be non-instance member
        # variables.
        BraxOptimizer.optimizer = optax.sgd(self.model.config['lr'])
        BraxOptimizer.model = self.model
        BraxOptimizer.loss_func = self.loss_func
        self.optimizer_state = None

    @staticmethod
    @jax.jit
    def cls_compute_loss_and_grads(opt_dev, root_state, seed_final, seed_act):
        def forward_and_get_loss(delta):
            mut_act = BraxOptimizer.model.mutate_seed_action(
                delta, seed_act, BraxOptimizer.model.config.mut_steps)
            opt_final = BraxOptimizer.model.step(root_state, mut_act)
            return BraxOptimizer.loss_func.apply(opt_final, seed_final)

        loss_value, grads = jax.value_and_grad(forward_and_get_loss)(opt_dev)
        return loss_value, grads

    @staticmethod
    @jax.jit
    def cls_update_opt_dev(opt_dev, grads, optimizer_state):
        updates, opt_state = BraxOptimizer.optimizer.update(
            grads, optimizer_state, opt_dev)
        opt_dev = optax.apply_updates(opt_dev, updates)
        return opt_dev, opt_state

    def compute_loss_and_grads(self, opt_dev, root_state, seed_final, seed_act):
        return self.cls_compute_loss_and_grads(opt_dev, root_state,
                                               seed_final, seed_act)

    def update_params(self, opt_dev, grads):
        opt_dev, self.optimizer_state = self.cls_update_opt_dev(
            opt_dev, grads, self.optimizer_state)
        return opt_dev

    def before_optimization(self, mut_dev, root_state, seed_final, seed_act):
        self.optimizer_state = BraxOptimizer.optimizer.init(mut_dev)
