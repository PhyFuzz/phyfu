from jax import numpy as jnp
from omegaconf import OmegaConf

from phyfu.utils.printer_utils import to_readable_mapping, list_to_str
from phyfu.utils.path_utils import ModulePath

from phyfu.common.registry import Registry

from phyfu.brax_mutate import model_loader
from phyfu.array_utils.jax_array import JaxArrayUtils
from phyfu.brax_mutate.loss_func import BraxLossFunction
from phyfu.brax_mutate.optimizer import BraxOptimizer
from phyfu.brax_mutate import bug_oracle


class BraxRegistry:
    factory = {
        "two_balls": {
            "fuzz": model_loader.TwoBalls,
            "find_errors": bug_oracle.TwoBallsOracle
        },
        "ur5e": {
            "fuzz": model_loader.UR5E,
            "find_errors": bug_oracle.UR5EOracle
        }
    }

    @staticmethod
    def register(cfg):
        Registry.module_path_utils = ModulePath("brax", cfg.model_name)
        Registry.array_utils = JaxArrayUtils()

        if cfg.operation == "fuzz" or cfg.operation == "both":
            Registry.model = BraxRegistry.factory[cfg.model_name]["fuzz"](
                Registry.module_path_utils, Registry.array_utils, cfg.get("extra_opts", None))
            Registry.loss_func = BraxLossFunction(Registry.model.config.loss_func)
            Registry.optimizer = BraxOptimizer(
                Registry.model, Registry.loss_func, Registry.array_utils)
        oracle_cfg = OmegaConf.load(Registry.module_path_utils.analysis_config_path)
        if cfg.get("extra_opts", None):
            oracle_cfg = OmegaConf.merge(oracle_cfg, cfg.extra_opts)
        Registry.bug_oracle = \
            BraxRegistry.factory[cfg.model_name]["find_errors"](oracle_cfg)

        to_readable_mapping[jnp.array([1.]).__class__] = list_to_str
