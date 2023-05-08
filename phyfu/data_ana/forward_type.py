from omegaconf import OmegaConf, ListConfig
import numpy as np

from phyfu.common.registry import Registry
from phyfu.utils.analysis_utils import get_time_stamp_from_summary, find_matching_config, \
    update_result
from phyfu.data_ana.get_pos_list import get_pos_list
from phyfu.utils.path_utils import LogPathUtils, PV_THRESHOLD_PATH, CAUSE_RESULTS_PATH
from phyfu.utils.log_utils import MetaInfo


def build_dev_list(time_stamp, p_dev_list, v_dev_list, sel_ids, model, mp, au):
    _, forward_err_list = get_pos_list(mp, time_stamp)
    log_path_utils = LogPathUtils(mp, False, time_stamp)
    for test_time in forward_err_list:
        meta_info: MetaInfo = au.load(log_path_utils.meta_info_per_opt_path(test_time))[0]
        seed_trace = model.step_trace(
            meta_info.root_state, meta_info.seed_action, sel_ids)
        mut_act = model.mutate_seed_action(meta_info.mut_dev_after,
                                           meta_info.seed_action,
                                           model.config.mut_steps)
        mut_trace = model.step_trace(meta_info.root_state, mut_act, sel_ids)
        trace_diff = np.abs(seed_trace - mut_trace)
        dof = len(trace_diff[0]) // 2
        p_diff = trace_diff[:, :dof]
        v_diff = trace_diff[:, dof:]
        p_dev = np.std(p_diff)
        v_dev = np.std(v_diff)
        p_dev_list.append(p_dev)
        v_dev_list.append(v_dev)


def classify_forward_errors():
    pv_threshold_list: ListConfig = OmegaConf.load(PV_THRESHOLD_PATH)

    model = Registry.model

    mp = Registry.module_path_utils
    au = Registry.array_utils

    pv_threshold = find_matching_config(mp, pv_threshold_list)

    sel_ids = set(range(model.config.num_steps))
    p_dev_list, v_dev_list = [], []

    if mp.model_name == "mpm":
        for ts in mp.get_all_time_stamps():
            build_dev_list(ts, p_dev_list, v_dev_list, sel_ids, model, mp, au)
    else:
        ts = get_time_stamp_from_summary(mp.module, mp.model_name)
        build_dev_list(ts, p_dev_list, v_dev_list, sel_ids, model, mp, au)

    p_dev_list = np.array(p_dev_list)
    p_dev_dev = np.std(p_dev_list)
    p_dev_median = np.median(p_dev_list)

    v_dev_list = np.array(v_dev_list)
    v_dev_dev = np.std(v_dev_list)
    v_dev_median = np.median(v_dev_list)

    n_p_err = sum(1 for p_dev in p_dev_list
                  if p_dev > p_dev_median - p_dev_dev * pv_threshold['pt'])
    n_v_err = sum(1 for v_dev in v_dev_list
                  if v_dev > v_dev_median - v_dev_dev * pv_threshold['vt'])
    print("#Position errors:", n_p_err)
    print("#Velocity errors:", n_v_err)

    update_result(CAUSE_RESULTS_PATH, mp, {"position_err": n_p_err, "velocity_err": n_v_err})
