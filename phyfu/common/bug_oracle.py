from abc import ABC, abstractmethod
from phyfu.utils.log_utils import MetaInfo


class BugOracle(ABC):
    def __init__(self, oracle_cfg):
        self.oracle_cfg = oracle_cfg

    def is_loss_too_large_meta_info(self, meta_info: MetaInfo):
        return meta_info.min_loss > self.oracle_cfg.min_loss_threshold

    def is_loss_too_large_raw(self, min_loss):
        return min_loss > self.oracle_cfg.min_loss_threshold

    def is_deviated_meta_info(self, meta_info: MetaInfo):
        diff_before, diff_after = self.get_diff(
            meta_info.mut_init_before, meta_info.mut_init_after, meta_info.seed_init)
        return self.is_deviated(diff_before, diff_after)

    def is_deviated_raw(self, mut_init_before, mut_init_after, seed_init):
        diff_before, diff_after = self.get_diff(mut_init_before, mut_init_after, seed_init)
        return self.is_deviated(diff_before, diff_after)

    @abstractmethod
    def get_diff(self, mut_init_before, mut_init_after, seed_init):
        """
        Get the mutant's difference with the seed before and after optimization.

        :param mut_init_before: the mutant's initial state on the simulation trace
         before optimization.
        :param mut_init_after: the mutant's initial state on simulation trace
         after optimization.
        :param seed_init: the seed's state at the beginning of simulation trace.
        :return: diff_before, diff_after

        """

    @abstractmethod
    def is_deviated(self, diff_before, diff_after):
        """
        Check whether the given diff pairs are buggy or not

        :param diff_before: the difference between mutant's initial state
         before optimization and seed initial state.
        :param diff_after: the difference between mutant's initial
        state after optimization and the seed initial state.
        :return: True if considered buggy else False

        """
