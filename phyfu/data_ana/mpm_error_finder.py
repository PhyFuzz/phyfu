import os

from phyfu.array_utils.array_interface import ArrayUtils
from phyfu.utils.path_utils import ModulePath, LogPathUtils
from phyfu.utils.log_utils import DataAnalysisLogger
from phyfu.common.bug_oracle import BugOracle
from phyfu.data_ana.error_finder import check


def set_module_path_root(module_path_utils: ModulePath, bug_checker: BugOracle):
    extra_paths = bug_checker.oracle_cfg.get("extra_paths")
    if isinstance(extra_paths, list) and isinstance(extra_paths[0], str):
        module_path_utils.results_dir = os.path.join(
            module_path_utils.results_dir, *extra_paths)
    elif isinstance(extra_paths, str):
        module_path_utils.results_dir = os.path.join(
            module_path_utils.results_dir, extra_paths)


def check_all(module_path_utils: ModulePath, array_utils: ArrayUtils, bug_checker: BugOracle):
    set_module_path_root(module_path_utils, bug_checker)

    total_loss, total_dev, total_crash = 0, 0, 0

    for time_stamp in module_path_utils.get_all_time_stamps():
        print('------------------')
        print(time_stamp)

        bug_checker.oracle_cfg.time_stamp = time_stamp
        check(module_path_utils, array_utils, bug_checker)
        log_path_utils = LogPathUtils(module_path_utils, False, time_stamp)
        n_loss, n_dev, n_crashes = DataAnalysisLogger().load(log_path_utils)

        total_dev += n_dev
        total_loss += n_loss
        total_crash += n_crashes

    print("#Total loss too large:", total_loss)
    print("#Total dev", total_dev)
    print("#Total crashes:", total_crash)
