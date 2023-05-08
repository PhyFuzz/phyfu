from typing import List, Union
from phyfu.utils.path_utils import LogPathUtils, ModulePath


def get_pos_list(module_path_utils: ModulePath, time_stamp: Union[List[str], str]):
    """
    Return the test_time of positives
    :param module_path_utils: module path
    :param time_stamp: the time stamp to be checked
    :return: loss_too_large_list, non_ident_init_list
    """

    path_handler = LogPathUtils(module_path_utils, False, time_stamp)

    with open(path_handler.data_analysis_path, 'r') as f:
        lines = f.readlines()

    loss_large, non_identical = [], []

    for line in lines:
        if line.startswith("test_time"):
            test_time = int(line[11:line.index(",")])
            if "identical" in line:
                non_identical.append(test_time)
            else:
                loss_large.append(test_time)

    return loss_large, non_identical
