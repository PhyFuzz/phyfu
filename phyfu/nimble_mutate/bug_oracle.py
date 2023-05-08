import torch

from phyfu.common.bug_oracle import BugOracle


def clip_diff(a: torch.Tensor, threshold):
    return torch.where(a > threshold, a, 0.0)


class NimbleOracle(BugOracle):

    def get_diff(self, mut_init_before: torch.Tensor, mut_init_after: torch.Tensor,
                 seed_init: torch.Tensor):
        diff_before = torch.abs(mut_init_before - seed_init)
        diff_after = clip_diff(torch.abs(mut_init_after - seed_init),
                               self.oracle_cfg.diff_tolerance)
        return diff_before, diff_after

    def is_deviated(self, diff_before, diff_after):
        return torch.any(diff_after > self.oracle_cfg.sigma * diff_before)