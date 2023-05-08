import os.path
from typing import Dict
from omegaconf import OmegaConf, ListConfig, DictConfig

from phyfu.utils.path_utils import ModulePath, TIME_STAMPS_PATH


def get_time_stamp_from_summary(module, model_name):
    return find_matching_config(
        ModulePath(module, model_name), OmegaConf.load(TIME_STAMPS_PATH)).art_time_stamp


def find_matching_config(module_path_utils: ModulePath,
                         summary: ListConfig) -> DictConfig:
    flag = False
    for s in summary:
        if s.module == module_path_utils.module and \
                s.model_name == module_path_utils.model_name:
            flag = True
            break

    if not flag:
        raise RuntimeError(f"Cannot find the specified configurations: "
                           f"{module_path_utils.module}, {module_path_utils.model_name}")
    return s


def create_if_no_matching(module_path_utils: ModulePath,
                          existing_list: ListConfig) -> DictConfig:
    try:
        matched = find_matching_config(
            module_path_utils, existing_list)
    except RuntimeError:
        matched = OmegaConf.create({"module": module_path_utils.module,
                                    "model_name": module_path_utils.model_name})
        existing_list.append(matched)
        matched = existing_list[-1]
    return matched


def load_list_config(file_path) -> ListConfig:
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            f.write("")
        results: ListConfig = OmegaConf.create([])
    else:
        results: ListConfig = OmegaConf.load(file_path)
    return results


def update_result(result_file, module_path_utils: ModulePath, update_dict: Dict):
    result_list = load_list_config(result_file)
    try:
        result = find_matching_config(module_path_utils, result_list)
    except RuntimeError:
        result = OmegaConf.create({
            "module": module_path_utils.module, "model_name": module_path_utils.model_name,
            **update_dict
        })
        result_list.append(result)
    else:
        for k, v in update_dict.items():
            result[k] = v
    OmegaConf.save(result_list, result_file)
