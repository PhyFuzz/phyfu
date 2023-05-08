import os
import random
import math
from typing import List
import time

from phyfu.array_utils.array_interface import ArrayUtils
from phyfu.common.model_loader import Model
from phyfu.utils.log_utils import MetaInfo
from phyfu.utils.path_utils import LogPathUtils, FINDINGS_PATH
from phyfu.common.registry import Registry
from phyfu.common.art_seed_getter import ArtSeedGetter
from phyfu.utils.analysis_utils import get_time_stamp_from_summary, update_result


class MockedArtSeedGetter(ArtSeedGetter):
    def __init__(self, au: ArrayUtils, model: Model, test_time_info_list):
        super().__init__(au, model)
        self._cur_idx = 0
        self.test_time_info_list = test_time_info_list

    @property
    def cur_idx(self):
        return self._cur_idx

    @cur_idx.setter
    def cur_idx(self, idx):
        self._cur_idx = idx

    def generate_points(self, n):
        if self.cur_idx + n > len(self.test_time_info_list):
            return [a.root_state for a in random.choices(self.test_time_info_list, k=n)], \
                list(range(n))
        else:
            return [a.root_state for a in
                    self.test_time_info_list[self.cur_idx:self.cur_idx + n]], list(range(n))


def get_scheduling_overhead():
    mp = Registry.module_path_utils
    au = Registry.array_utils
    bug_checker = Registry.bug_oracle

    art_time_stamp = get_time_stamp_from_summary(mp.module, mp.model_name)
    log_path_utils = LogPathUtils(mp, False, art_time_stamp)

    test_time_info_list: List[MetaInfo] = []

    sorted_test_time_numbers = [int(d) for d in log_path_utils.get_all_test_time_dirs()]
    sorted_test_time_numbers.sort()

    for i in sorted_test_time_numbers:
        meta_info_path = log_path_utils.meta_info_per_opt_path(i)
        if not os.path.exists(meta_info_path): # crash
            # Assume no crash at the first test round
            test_time_info_list.append(test_time_info)
            continue
        test_time_info = au.load(meta_info_path)[0]
        test_time_info_list.append(test_time_info)

    seed_getter = MockedArtSeedGetter(au, Registry.model, test_time_info_list)

    # Warmup
    for _ in range(2):
        bug_checker.is_loss_too_large_meta_info(test_time_info_list[0])
        bug_checker.is_deviated_meta_info(test_time_info_list[0])


    start_time = time.time()
    for cur_idx, test_time_info in enumerate(test_time_info_list):
        seed_getter.cur_idx = cur_idx
        seed_getter.guided_gen_seed()

        if bug_checker.is_loss_too_large_meta_info(test_time_info) or\
            bug_checker.is_deviated_meta_info(test_time_info):
            seed_getter.add_executed_item(test_time_info.root_state, True)
        else:
            seed_getter.add_executed_item(test_time_info.root_state, False)
    end_time = time.time()

    t = end_time - start_time
    minutes = math.ceil(t / 60)

    print(f"Overhead of seed scheduling for {mp.module} {mp.model_name}: {minutes} minutes")
    update_result(FINDINGS_PATH, mp, {"scheduling_overhead": minutes})
