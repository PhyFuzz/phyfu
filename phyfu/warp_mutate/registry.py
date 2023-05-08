import numpy as np
from omegaconf import OmegaConf

from phyfu.common.registry import Registry
from phyfu.utils.path_utils import ModulePath
from phyfu.utils.printer_utils import to_readable_mapping, list_to_str
from phyfu.array_utils.wp_array import WarpArrayUtils

import warp as wp
wp.init()

from warp.sim import State
from phyfu.warp_mutate import model_loader
from phyfu.warp_mutate.loss_func import SnakeLossFunc, TwoBallsLossFunc
from phyfu.warp_mutate.optimizer import WarpOpt
from phyfu.warp_mutate import bug_oracle


class WarpRegistry:
    factory = {
        "snake": {
            "fuzz": model_loader.SnakeModel,
            "loss_func": SnakeLossFunc,
            "readable_mapping": lambda s: f"{s.body_q.to('cpu').numpy()[:, [0,1,5,6]]}\n"
                                          f"{s.body_qd.to('cpu').numpy()[:, 2:5]}",
            "find_errors": bug_oracle.SnakeOracle
        },
        "two_balls": {
            "fuzz": model_loader.TwoBalls,
            "loss_func": TwoBallsLossFunc,
            "readable_mapping": lambda s: f"{s.particle_q.to('cpu').numpy()[:, 0]}\n"
                                          f"{s.particle_qd.to('cpu').numpy()[:, 0]}",
            "find_errors": bug_oracle.TwoBallsOracle
        }
    }

    @staticmethod
    def register(cfg):
        np.set_printoptions(precision=2)
        Registry.module_path_utils = ModulePath("warp", cfg.model_name)
        Registry.array_utils = WarpArrayUtils()
        if cfg.operation == "fuzz" or cfg.operation == "both":
            Registry.model = WarpRegistry.factory[cfg.model_name]["fuzz"](
                Registry.module_path_utils, Registry.array_utils, cfg.get("extra_opts", None))
            Registry.loss_func = WarpRegistry.factory[cfg.model_name]["loss_func"](
                Registry.model.config.loss_func)
            Registry.optimizer = WarpOpt(
                Registry.model, Registry.loss_func, Registry.array_utils)
        oracle_cfg = OmegaConf.load(Registry.module_path_utils.analysis_config_path)
        if cfg.get("extra_opts", None):
            oracle_cfg = OmegaConf.merge(oracle_cfg, cfg.extra_opts)
        Registry.bug_oracle = \
            WarpRegistry.factory[cfg.model_name]["find_errors"](oracle_cfg)

        to_readable_mapping.update({
            np.array([1.0], dtype=np.float32).__class__: lambda x: list_to_str(x.flatten())
        })
        to_readable_mapping.update({
            State: WarpRegistry.factory[cfg.model_name]["readable_mapping"]
        })
