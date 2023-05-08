import torch
from omegaconf import OmegaConf

from phyfu.common.registry import Registry
from phyfu.utils.path_utils import ModulePath
from phyfu.utils.printer_utils import to_readable_mapping, list_to_str
from phyfu.array_utils.torch_array import TorchArrayUtils

from phyfu.nimble_mutate import model_loader, bug_oracle
from phyfu.nimble_mutate.loss_func import NimbleLossFunc
from phyfu.nimble_mutate.optimizer import NimbleOpt
from phyfu.nimble_mutate.atlas import Atlas


class NimbleRegistry:
    factory = {
        "two_balls": {
            "fuzz": model_loader.TwoBalls,
            "find_errors": bug_oracle.NimbleOracle
        },
        "catapult": {
            "fuzz": model_loader.Catapult,
            "find_errors": bug_oracle.NimbleOracle
        }
    }

    @staticmethod
    def register(cfg):
        Registry.module_path_utils = ModulePath("nimble", cfg["model_name"])
        Registry.array_utils = TorchArrayUtils()
        if cfg["operation"] == "fuzz" or cfg["operation"] == "both":
            Registry.model = NimbleRegistry.factory[cfg["model_name"]]["fuzz"](
                Registry.module_path_utils, Registry.array_utils, cfg.get("extra_opts", None))
            Registry.loss_func = NimbleLossFunc(Registry.model.config.loss_func)
            Registry.optimizer = NimbleOpt(
                Registry.model, Registry.loss_func, Registry.array_utils)
        oracle_cfg = OmegaConf.load(Registry.module_path_utils.analysis_config_path)
        if cfg.get("extra_opts", None):
            oracle_cfg = OmegaConf.merge(oracle_cfg, cfg.extra_opts)
        Registry.bug_oracle = \
            NimbleRegistry.factory[cfg["model_name"]]["find_errors"](oracle_cfg)

        to_readable_mapping.update({
            torch.Tensor([1.0]).__class__: lambda x: list_to_str(x.flatten())
        })
