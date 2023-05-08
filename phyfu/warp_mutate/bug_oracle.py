from abc import ABC, abstractmethod
import numpy as np
import quaternion

from phyfu.common.bug_oracle import BugOracle
from phyfu.array_utils.wp_array import NpState


def euclidian_dist(x: np.ndarray, y: np.ndarray):
    return np.linalg.norm(x - y, axis=1)

def angle_dist(x: np.ndarray, y: np.ndarray):
    return np.array([
        np.multiply(np.quaternion(q1[0], q1[1], q1[2], q1[3]),
                    np.quaternion.conjugate(np.quaternion(q2[0], q2[1], q2[2], q2[3])))
        for q1, q2 in zip(x, y)])


def clip_diff_after(diff_after, threshold):
    return np.where(diff_after < threshold, 0.0, diff_after)


class WarpOracle(BugOracle, ABC):
    @staticmethod
    @abstractmethod
    def state_diff(s1: NpState, s2: NpState):
        ...

    def get_diff(self, mut_init_before, mut_init_after, seed_init):
        diff_before = self.state_diff(mut_init_before, seed_init)
        diff_after = clip_diff_after(self.state_diff(mut_init_after, seed_init),
                                     self.oracle_cfg.diff_tolerance)
        return diff_before, diff_after


class SnakeOracle(WarpOracle):
    def is_deviated(self, diff_before, diff_after):
        return np.sum(diff_after > self.oracle_cfg.sigma * diff_before) > \
                self.oracle_cfg.cnt_tolerance

    @staticmethod
    def state_diff(s1: NpState, s2: NpState):
        if not isinstance(s1, NpState):
            s1, s2 = NpState.from_wp_state(s1), NpState.from_wp_state(s2)
        return np.concatenate([
            euclidian_dist(s1.q[:, :3], s2.q[:, :3]),
            euclidian_dist(s1.q[:, 3:], s2.q[:, 3:]),
            euclidian_dist(s1.qd[:, :3], s2.qd[:, :3]),
            euclidian_dist(s1.qd[:, 3:], s2.qd[:, 3:]),
        ])


class TwoBallsOracle(WarpOracle):
    @staticmethod
    def state_diff(s1: NpState, s2: NpState):
        if not isinstance(s1, NpState):
            s1, s2 = NpState.from_wp_state(s1), NpState.from_wp_state(s2)
        return np.concatenate([
            np.abs(s1.q[:, 0] - s2.q[:, 0]).flatten(),
            np.abs(s1.qd[:, 0] - s2.qd[:, 0]).flatten(),
        ])

    def is_deviated(self, diff_before, diff_after):
        return np.any(diff_after > self.oracle_cfg.sigma * diff_before)