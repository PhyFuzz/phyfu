import os
from omegaconf import OmegaConf
from time import sleep
import numpy as np

from phyfu.common.registry import Registry
from phyfu.utils.log_utils import cur_time
from phyfu.common.seed_getter import SeedGetter
from phyfu.common.art_seed_getter import ArtSeedGetter
from phyfu.common.fuzzer import Fuzzer
from phyfu.common.guided_fuzzing import GuidedFuzzer
from phyfu.common.mutate import Mutator
from phyfu.utils.log_utils import FuzzingLogger
from phyfu.utils.path_utils import TIME_STAMPS_PATH
from phyfu.utils.analysis_utils import update_result


def fuzz(label=None):
    mutator = Mutator(Registry.model, Registry.array_utils)
    optimizer = Registry.optimizer
    while os.path.exists(os.path.join(Registry.module_path_utils.results_dir, cur_time())):
        sleep(np.random.randint(1, 11))
    logger = FuzzingLogger(disable=Registry.model.config.disable_logging,
                           module_path_utils=Registry.module_path_utils,
                           array_utils=Registry.array_utils, label=label)
    mutate_cfg = Registry.model.config
    if not mutate_cfg.disable_logging:
        log_path_handler = logger.path_handler
        OmegaConf.save(Registry.model.config,
                       os.path.join(log_path_handler.log_root, "mutate.yaml"))
        if mutate_cfg.seed_getter.type == "art":
            OmegaConf.save(Registry.bug_oracle.oracle_cfg,
                           os.path.join(log_path_handler.log_root, "analysis.yaml"))
    if Registry.model.config.seed_getter.type == "random":
        seed_getter = SeedGetter(Registry.array_utils, Registry.model)
        fuzzer = Fuzzer(
            Registry.model, Registry.loss_func, seed_getter, Registry.array_utils,
            mutator, optimizer, logger)
    elif Registry.model.config.seed_getter.type == "art":
        seed_getter = ArtSeedGetter(Registry.array_utils, Registry.model)
        fuzzer = GuidedFuzzer(
            Registry.model, Registry.loss_func, seed_getter, Registry.array_utils,
            mutator, optimizer, Registry.bug_oracle, logger)
    else:
        raise RuntimeError(f"Invalid option in configs/fuzzing/{Registry.model.config.module}/"
                           f"{Registry.model.config.model_name}/mutate.yaml: "
                           f"seed_getter.type: {Registry.model.config.seed_getter.type}. "
                           f"Should be one of 'art' or 'random'.")
    fuzzer.fuzz()

    if not mutate_cfg.disable_logging:
        update_result(TIME_STAMPS_PATH, Registry.module_path_utils,
                      {f"{Registry.model.config.seed_getter.type}_time_stamp":
                           logger.time_stamp})
        return logger.time_stamp
