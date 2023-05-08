import os.path
from typing import List

from phyfu.array_utils.array_interface import ArrayUtils
from phyfu.common.bug_oracle import BugOracle
from phyfu.utils.log_utils import MetaInfo, DataAnalysisLogger
from phyfu.utils.path_utils import LogPathUtils, ModulePath
from phyfu.utils.printer_utils import EnhancedPrinter, to_readable


def check(module_path_utils: ModulePath, array_utils: ArrayUtils, bug_checker: BugOracle):
    oracle_cfg = bug_checker.oracle_cfg
    log_path_utils = LogPathUtils(module_path_utils, False, oracle_cfg.time_stamp)

    printer = EnhancedPrinter(to_file=oracle_cfg.write_to_file, print_out=False,
                              target_file=log_path_utils.data_analysis_path)

    if os.path.exists(log_path_utils.meta_arr_path):
        meta_info_list: List[MetaInfo] = array_utils.load(log_path_utils.meta_arr_path)

    n_test_times = len([d for d in os.listdir(log_path_utils.log_root)
                        if os.path.isdir(os.path.join(log_path_utils.log_root, str(d)))])

    num_loss_too_large, num_deviated_init_state, n_crashes = 0, 0, 0

    for i in range(n_test_times):
        test_time = i + 1
        # readable_info = readable_info_list[i]
        if os.path.exists(log_path_utils.meta_arr_path):
            arr_info = meta_info_list[i]
        else:
            if not os.path.exists(log_path_utils.meta_info_per_opt_path(test_time)):
                n_crashes += 1
                continue
            arr_info = array_utils.load(log_path_utils.meta_info_per_opt_path(test_time))[0]
        if arr_info.min_loss > oracle_cfg.min_loss_threshold:
            printer.print("=====================================")
            printer.print(f"test_time: {test_time}, "
                          f"loss too large: {to_readable(arr_info.min_loss)}")
            printer.print(f"mut_init_before: {to_readable(arr_info.mut_init_before)}\n"
                          f"mut_init_after: {to_readable(arr_info.mut_init_after)}")
            num_loss_too_large += 1
            continue
        diff_before, diff_after = bug_checker.get_diff(
            arr_info.mut_init_before, arr_info.mut_init_after, arr_info.seed_init)
        if bug_checker.is_deviated(diff_before, diff_after):
            printer.print("=====================================")
            printer.print(f"test_time: {test_time}, init not identical")
            printer.print("diff after: ", to_readable(diff_after))
            printer.print("diff before:", to_readable(diff_before))
            # Avoid ugly output of nan on 0. / 0.
            printer.print("sigma:      ", to_readable(diff_after / (diff_before + 1e-7)))
            num_deviated_init_state += 1

    # Caution: Calling printer.close() is a must!

    printer.print("=======================")
    printer.print(DataAnalysisLogger().to_summary(
        num_loss_too_large, num_deviated_init_state, n_crashes), print_out=True)
    printer.close()
