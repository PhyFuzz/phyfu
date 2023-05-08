import torch

from phyfu.common.optimizer import Optimizer


class NimbleOpt(Optimizer):
    def before_optimization(self, mut_dev, root_state, seed_final, seed_act):
        mut_dev.requires_grad_(True)
        self.optimizer = torch.optim.Adam([mut_dev], lr=self.model.config.lr)

    def compute_loss_and_grads(self, opt_dev, root_state, seed_final, seed_act):
        opt_dev = self.optimizer.param_groups[0]['params'][0]
        mut_act = self.model.mutate_seed_action(
            opt_dev, seed_act, self.model.config.mut_steps)
        mut_final = self.model.step(root_state, mut_act)
        loss = self.loss_func.apply(mut_final, seed_final)
        loss.backward()
        grads = torch.clone(opt_dev.grad)
        return loss.item(), grads

    def update_params(self, opt_dev, grads):
        with torch.no_grad():
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
        opt_dev = self.optimizer.param_groups[0]['params'][0].clone()
        return opt_dev

