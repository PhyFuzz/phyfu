from phyfu.common.optimizer import Optimizer
from phyfu.adam.adam import Adam


class TaichiOptimizer(Optimizer):
    def before_optimization(self, mut_dev, root_state, seed_final, seed_act):
        self.optimizer = Adam(lr=self.model.config.lr)
        self.model.before_opt(root_state, seed_final, seed_act)

    def compute_loss_and_grads(self, opt_dev, root_state, seed_final, seed_act):
        loss, grads = self.model.loss_and_grads(opt_dev)
        if str(loss).lower() == 'nan':
            raise RuntimeError("Encountered NaN during optimization")
        return loss, grads

    def update_params(self, opt_dev, grads):
        opt_dev = self.optimizer.update(opt_dev, grads, param_name="opt_dev")
        return opt_dev
