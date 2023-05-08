import warp as wp
import warp.sim
from warp.sim import State
import numpy as np
from dataclasses import dataclass

from phyfu.array_utils.np_array import NpArrayUtils

from phyfu.utils.log_utils import MetaInfo


@dataclass
class NpState:
    is_body: bool
    q: np.ndarray
    qd: np.ndarray

    def __str__(self):
        if not self.is_body:
            return f"{[self.q[0][0], self.q[1][0], self.qd[0][0], self.qd[1][0]]}"
        else:
            return f"position: {self.q}\nvelocity: {self.qd}"

    @staticmethod
    def from_wp_state(s: State):
        if s.body_count:
            return NpState(
                True,
                s.body_q.to("cpu").numpy(),
                s.body_qd.to("cpu").numpy()
            )
        else:
            return NpState(
                False,
                s.particle_q.to("cpu").numpy(),
                s.particle_qd.to("cpu").numpy()
            )

    def to_wp_state(self):
        s = State()
        if self.is_body:
            s.body_count = 1
            s.body_q = wp.from_numpy(self.q, dtype=wp.float32, requires_grad=True)
            s.body_qd = wp.from_numpy(self.qd, dtype=wp.float32, requires_grad=True)
            s.body_f = wp.zeros_like(s.body_qd, requires_grad=True)
        else:
            s.particle_count = 1
            s.particle_q = wp.from_numpy(self.q, dtype=wp.vec3, requires_grad=True)
            s.particle_qd = wp.from_numpy(self.qd, dtype=wp.vec3, requires_grad=True)
            s.particle_f = wp.zeros_like(s.particle_qd, requires_grad=True)

        return s


class WarpArrayUtils(NpArrayUtils):
    @staticmethod
    def loss_to_float(loss) -> float:
        if isinstance(loss, np.ndarray):
            return NpArrayUtils.loss_to_float(loss)
        return loss.to("cpu").numpy().item()

    @staticmethod
    def save(file, arr):
        if (isinstance(arr, list) or isinstance(arr, tuple)) and isinstance(arr[0], MetaInfo):
            for item in arr:
                item.root_state = NpState.from_wp_state(item.root_state)
                item.seed_init = NpState.from_wp_state(item.seed_init)
                item.seed_final = NpState.from_wp_state(item.seed_final)
                item.mut_init_before = NpState.from_wp_state(item.mut_init_before)
                item.mut_init_after = NpState.from_wp_state(item.mut_init_after)
                item.mut_final_before = NpState.from_wp_state(item.mut_final_before)
                item.mut_final_after = NpState.from_wp_state(item.mut_final_after)
        NpArrayUtils.save(file, arr)

    # @staticmethod
    # def load(file):
    #     arr = NpArrayUtils.load(file)
    #     if (isinstance(arr, list) or isinstance(arr, tuple)) and isinstance(arr[0], MetaInfo):
    #         for item in arr:
    #             item.root_state = item.root_state.to_wp_state()
    #             item.seed_init = item.seed_init.to_wp_state()
    #             item.seed_final = item.seed_final.to_wp_state()
    #             item.mut_init_before = item.mut_init_before.to_wp_state()
    #             item.mut_init_after = item.mut_init_after.to_wp_state()
    #             item.mut_final_before = item.mut_final_before.to_wp_state()
    #             item.mut_final_after = item.mut_final_after.to_wp_state()
    #     return arr
