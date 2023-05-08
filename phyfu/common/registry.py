from dataclasses import dataclass
import numpy as np
from omegaconf import DictConfig, OmegaConf
from typing import Union

from phyfu.common.model_loader import Model
from phyfu.utils.path_utils import ModulePath
from phyfu.array_utils.array_interface import ArrayUtils
from phyfu.common.loss_func import LossFunction
from phyfu.common.optimizer import Optimizer
from phyfu.common.bug_oracle import BugOracle


@dataclass
class Registry:
    model: Model
    module_path_utils: ModulePath
    array_utils: ArrayUtils
    loss_func: LossFunction
    optimizer: Optimizer
    bug_oracle: BugOracle



def populate_registry(cfg: Union[DictConfig, dict]):
    if not isinstance(cfg, DictConfig):
        cfg = OmegaConf.create(cfg)
    np.random.seed(0)
    if cfg.module == "brax":
        from phyfu.brax_mutate.registry import BraxRegistry

        BraxRegistry.register(cfg)
    elif cfg.module == "taichi":
        from phyfu.taichi_mutate.registry import TaichiRegistry

        TaichiRegistry.register(cfg)
    elif cfg.module == "warp":
        from phyfu.warp_mutate.registry import WarpRegistry

        WarpRegistry.register(cfg)
    elif cfg.module == "nimble":
        from phyfu.nimble_mutate.registry import NimbleRegistry

        NimbleRegistry.register(cfg)
