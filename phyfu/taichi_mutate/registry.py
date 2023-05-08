import os.path

import numpy as np
from omegaconf import OmegaConf

from phyfu.common.registry import Registry
from phyfu.utils.path_utils import ModulePath
from phyfu.utils.printer_utils import to_readable_mapping, list_to_str

from phyfu.taichi_mutate import model_loader
from phyfu.taichi_mutate import bug_oracle
from phyfu.array_utils.np_array import NpArrayUtils
from phyfu.taichi_mutate.loss_func import TaichiLossFunc
from phyfu.taichi_mutate.optimizer import TaichiOptimizer


class TaichiRegistry:
    factory = {
        "two_balls": {
            "fuzz": model_loader.TwoBallsModel,
            "find_errors": bug_oracle.TwoBallsOracle
        },
        "mpm": {
            "fuzz": model_loader.MpmModel,
            "find_errors": bug_oracle.MpmOracle
        }
    }

    @staticmethod
    def register(cfg):
        Registry.module_path_utils = ModulePath("taichi", cfg.model_name)
        Registry.array_utils = NpArrayUtils()
        oracle_cfg = OmegaConf.load(Registry.module_path_utils.analysis_config_path)
        if cfg.get("extra_opts", None):
            oracle_cfg = OmegaConf.merge(oracle_cfg, cfg.extra_opts)
        Registry.bug_oracle = \
            TaichiRegistry.factory[cfg.model_name]["find_errors"](oracle_cfg)
        if cfg.operation == "fuzz" or cfg.operation == "both":
            Registry.model = TaichiRegistry.factory[cfg.model_name]["fuzz"](
                Registry.module_path_utils, Registry.array_utils, cfg.get("extra_opts", None))
            Registry.loss_func = TaichiLossFunc(Registry.model.config.loss_func)
            Registry.optimizer = TaichiOptimizer(
                Registry.model, Registry.loss_func, Registry.array_utils)

            if cfg.model_name == "mpm":
                if cfg.operation == "both" and Registry.model.config.seed_getter.type \
                    != Registry.bug_oracle.oracle_cfg.extra_paths:
                    raise RuntimeError(f"Incompatible extra paths between fuzz seed_getter: "
                                       f"{Registry.model.config.seed_getter.type} and "
                                       f"the extra_paths param in oracle config: "
                                       f"{Registry.bug_oracle.oracle_cfg.extra_paths}!")
                Registry.module_path_utils.results_dir = os.path.join(
                    Registry.module_path_utils.results_dir,
                    Registry.model.config.seed_getter.type
                )
        elif cfg.model_name == "mpm":
            Registry.module_path_utils.results_dir = os.path.join(
                Registry.module_path_utils.results_dir,
                Registry.bug_oracle.oracle_cfg.extra_paths
            )

        to_readable_mapping.update({
            np.array([1.0], dtype=np.float32).__class__: lambda x: list_to_str(x.flatten())
        })