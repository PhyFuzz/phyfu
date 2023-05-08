from abc import ABC
import numpy as np

from phyfu.common.bug_oracle import BugOracle
from phyfu.taichi_mutate.model_loader import QP


def clip_diff_after(diff_after, threshold):
    return np.where(diff_after < threshold, 0.0, diff_after)


def abs_diff(q1: QP, q2: QP):
    return np.concatenate([np.abs(q1.x - q2.x).flatten(), np.abs(q1.v - q2.v).flatten()])


class TaichiOracle(BugOracle, ABC):
    def get_diff(self, mut_init_before: QP, mut_init_after: QP, seed_init: QP):
        diff_before = abs_diff(mut_init_before, seed_init)
        diff_after = clip_diff_after(abs_diff(mut_init_after, seed_init),
                                     self.oracle_cfg.diff_tolerance)
        return diff_before, diff_after


class TwoBallsOracle(TaichiOracle):
    def is_deviated(self, diff_before, diff_after):
        return np.any(diff_after > diff_before * self.oracle_cfg.sigma)


class MpmOracle(TaichiOracle):
    def is_deviated(self, diff_before, diff_after):
        # return np.any(diff_after > diff_before * self.oracle_cfg.sigma)
        return np.sum(diff_after > diff_before * self.oracle_cfg.sigma) \
            > self.oracle_cfg.cnt_tolerance
        # return np.sum(diff_after) > np.sum(diff_after) * self.oracle_cfg.sigma
