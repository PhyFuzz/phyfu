import numpy as np
import warp as wp
import warp.sim
from warp.sim import State

from phyfu.array_utils.array_interface import ArrayUtils
from phyfu.common.loss_func import LossFunction
from phyfu.common.optimizer import Optimizer
from phyfu.warp_mutate.model_loader import WarpModel, copy_src_to_tgt_state, clone_state
from phyfu.adam.adam import Adam


@wp.kernel
def add_mut(i: wp.int32, mut_steps: wp.int32,
            seed: wp.array(dtype=wp.float32), mut: wp.array(dtype=wp.float32),
            c: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    if i < mut_steps:
        c[tid] = seed[tid] * (1. + mut[tid])
    else:
        c[tid] = seed[tid]


class WarpOpt(Optimizer):
    def __init__(self, loader: WarpModel, loss_func: LossFunction, array_utils: ArrayUtils):
        super().__init__(loader, loss_func, array_utils)
        self.loader = loader
        self.model = loader.model
        self.loss_func = loss_func
        self.device = None

        self.substeps = loader.substeps
        self.sim_steps = loader.config.num_steps
        self.mut_steps = loader.config.mut_steps
        self.dt = loader.dt
        self.lr = loader.config.lr

        self.seed_act_list = [wp.empty(self.loader.dof_shape, dtype=wp.float32,
                device=self.device, requires_grad=False) for _ in range(self.sim_steps)]
        self.mut_act_list = [wp.empty(self.loader.dof_shape, dtype=wp.float32,
                                      device=self.device, requires_grad=True)
                             for _ in range(self.sim_steps)]
        self.mut_dev = wp.empty(self.loader.dof_shape, dtype=wp.float32,
                                device=self.device, requires_grad=True)
        self.seed_final = None
        self.graph = None
        self.tape = None

    def capture_graph(self):
        wp.capture_begin(device=self.device)
        self.tape = wp.Tape()
        with self.tape:
            for i, seed_act in enumerate(self.seed_act_list):
                wp.launch(add_mut, dim=(len(seed_act),),
                          inputs=[i, self.mut_steps, seed_act, self.mut_dev,
                                  self.mut_act_list[i]],
                          device=self.device)
                for j in range(self.substeps):
                    step_num = i * self.substeps + j
                    self.loader.states[step_num].clear_forces()
                    self.loader.integrator.simulate(
                        self.model, self.mut_act_list[i], self.loader.states[step_num],
                        self.loader.states[step_num + 1], self.dt)

            self.loss_func.apply(self.loader.states[self.sim_steps * self.substeps],
                                 self.seed_final)

            self.tape.backward(self.loss_func.loss)

        self.graph = wp.capture_end(device=self.device)

    def before_optimization(self, mut_dev: np.ndarray, root_state: State,
                            seed_final: State, seed_act_list: np.ndarray):
        self.mut_dev.assign(mut_dev)

        for i, a in enumerate(seed_act_list):
            self.seed_act_list[i].assign(a)

        if self.seed_final is None:
            self.seed_final = clone_state(seed_final)
        else:
            copy_src_to_tgt_state(seed_final, self.seed_final)

        self.loader.set_init_state(root_state)

        if self.graph is None:
            self.capture_graph()
        self.optimizer = Adam(lr=self.lr)

    def compute_loss_and_grads(self, opt_dev, root_state, seed_final, seed_act):
        self.mut_dev.assign(opt_dev)
        wp.capture_launch(self.graph)
        grads = self.mut_dev.grad.to("cpu").numpy()
        loss = self.loss_func.loss.to("cpu").numpy()
        self.tape.zero()
        return loss, grads

    def update_params(self, opt_dev, grads):
        opt_dev = self.optimizer.update(opt_dev, grads, param_name="opt_dev")
        return opt_dev
