import os
from omegaconf import OmegaConf, ListConfig

from phyfu.common.registry import Registry
from phyfu.utils.path_utils import ModulePath, LogPathUtils, TIME_STAMPS_PATH, FINDINGS_PATH
from phyfu.utils.log_utils import DataAnalysisLogger
from phyfu.data_ana.time_analysis import get_exec_time, seconds_to_readable
from phyfu.utils.analysis_utils import create_if_no_matching, load_list_config, \
    find_matching_config


def get_exec_info(al: DataAnalysisLogger,lp: LogPathUtils):
    n_loss, n_dev, n_crashes = al.load(lp)
    exec_time = get_exec_time(lp)
    return n_loss, n_dev, n_crashes, exec_time


def collect_info(case, finding, seed_getter, module_path_utils: ModulePath,
                 da: DataAnalysisLogger):
    module, model_name = module_path_utils.module, module_path_utils.model_name
    time_stamp_key = f"{seed_getter}_time_stamp"
    if not time_stamp_key in case:
        return
    time_stamp = case[time_stamp_key]
    lp = LogPathUtils(module_path_utils, False, time_stamp)
    if not os.path.exists(lp.data_analysis_path):
        print(f"Cannot find {module} {model_name} {time_stamp}, continuing")
        return
    n_loss, n_dev, n_crashes, total_time = get_exec_info(da, lp)
    n_total = n_dev + n_loss
    finding[f"{seed_getter}_err"] = n_total
    print(f"Total number of errors found by {seed_getter}:", n_total)
    print(f"Total number of crashes found by {seed_getter}:", n_crashes)
    if seed_getter == "art":
        finding['exec_time'] = seconds_to_readable(total_time)
        print("Total execution time:", finding['exec_time'])


def get_total_num_errors():
    time_stamp_list = OmegaConf.load(TIME_STAMPS_PATH)
    finding_list: ListConfig = load_list_config(FINDINGS_PATH)

    case = find_matching_config(Registry.module_path_utils, time_stamp_list)

    data_analysis_logger = DataAnalysisLogger()

    module, model_name = case.module, case.model_name
    module_path_utils = ModulePath(module, model_name)
    finding = create_if_no_matching(module_path_utils, finding_list)

    collect_info(case, finding, "art", module_path_utils, data_analysis_logger)
    collect_info(case, finding, "random", module_path_utils, data_analysis_logger)

    OmegaConf.save(finding_list, FINDINGS_PATH)
