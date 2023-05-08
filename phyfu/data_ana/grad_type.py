import numpy as np

from typing import List

from phyfu.utils.analysis_utils import get_time_stamp_from_summary, update_result
from phyfu.array_utils.np_array import NpArrayUtils
from phyfu.utils.path_utils import LogPathUtils, CAUSE_RESULTS_PATH
from phyfu.utils.log_utils import OptInfo
from phyfu.common.registry import Registry
from phyfu.data_ana.get_pos_list import get_pos_list


def inverse_num(array):
    n = len(array)
    if n <= 1:
        return 0, array

    mid = n // 2
    inverse_l, arr_l = inverse_num(array[:mid])
    inverse_r, arr_r = inverse_num(array[mid:])

    nl, nr = len(arr_l), len(arr_r)

    arr_l.append(float('inf'))
    arr_r.append(float('inf'))

    i, j = 0, 0
    new_arr = []
    inverse = inverse_l + inverse_r

    while i < nl or j < nr:
        if arr_l[i] <= arr_r[j]:
            inverse += j
            new_arr.append(arr_l[i])
            i += 1
        else:
            new_arr.append(arr_r[j])
            j += 1
    return inverse, new_arr


def is_grad_len_wrong(au, opt_info_list: List[OptInfo]):
    opt_len = np.array(
        [NpArrayUtils.length(au.to_numpy(opt_info_list[i].opt_dev))
         for i in range(1, len(opt_info_list))])
    grads_len = np.array(
        [NpArrayUtils.length(au.to_numpy(opt_info_list[i].grads))
         for i in range(1, len(opt_info_list))])

    sorted_ids = opt_len.argsort()
    grads_len = grads_len[sorted_ids]

    n = len(opt_len)

    n_err, _ = inverse_num(grads_len.tolist())

    t = n * 2 / 3

    return n_err > (t * (t - 1)) / 2


def is_grad_dir_wrong(au, opt_info_list: List[OptInfo]):
    n_dir_wrong_steps = 0
    for i in range(1, len(opt_info_list)):
        if au.angle(opt_info_list[i - 1].opt_dev, opt_info_list[i].grads) > \
                np.pi * 90 / 180:
            n_dir_wrong_steps += 1

    return n_dir_wrong_steps > (len(opt_info_list) - 1) // 2


def classify_single_test_time(au, loss_large_list, log_path_utils):
    dir_err_test_times = set()
    len_err_test_times = set()
    for test_time in loss_large_list:
        opt_info_list: List[OptInfo] = au.load(log_path_utils.opt_arr_path(test_time))
        try:
            if is_grad_dir_wrong(au, opt_info_list):
                dir_err_test_times.add(test_time)

            if is_grad_len_wrong(au, opt_info_list):
                len_err_test_times.add(test_time)
        except IndexError: # Could be a crash in MPM resulting NaN, so the guard would fail
            continue

    return dir_err_test_times, len_err_test_times


def get_single_error_distri(mp, au, time_stamp):
    loss_large_list, dev_list = get_pos_list(mp, time_stamp)
    log_path_utils = LogPathUtils(mp, False, time_stamp)
    Registry.bug_oracle.oracle_cfg.time_stamp = time_stamp
    dir_err_set, len_err_set = classify_single_test_time(au, loss_large_list, log_path_utils)
    unapparent_err_set = set(loss_large_list).difference(dir_err_set).difference(len_err_set)
    return len(dir_err_set), len(len_err_set), len(unapparent_err_set)


def classify_backward_errors():
    mp = Registry.module_path_utils
    au = Registry.array_utils
    n_dir_err, n_len_err, n_una_err = 0, 0, 0

    if mp.module == "taichi" and mp.model_name == "mpm":
        time_stamp_list = mp.get_all_time_stamps()
        for ts in time_stamp_list:
            nde, nle, nue = get_single_error_distri(mp, au, ts)
            n_dir_err += nde
            n_len_err += nle
            n_una_err += nue
    else:
        ts = get_time_stamp_from_summary(mp.module, mp.model_name)
        n_dir_err, n_len_err, n_una_err = get_single_error_distri(mp, au, ts)

    print("#Gradient direction errors:", n_dir_err)
    print("#Gradient extent errors:", n_len_err)
    print("#Unapparent errors:", n_una_err)

    update_result(CAUSE_RESULTS_PATH, mp,
                  {"grad_direction_err": n_dir_err, "grad_extent_err": n_len_err,
                   "unapparent_err": n_una_err})
