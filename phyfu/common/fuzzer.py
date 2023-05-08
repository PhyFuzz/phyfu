from phyfu.common.mutate import Mutator
from phyfu.common.seed_getter import SeedGetter
from phyfu.common.model_loader import Model
from phyfu.common.loss_func import LossFunction
from phyfu.common.optimizer import Optimizer

from phyfu.utils.timing_utils import TimingUtils
from phyfu.array_utils.array_interface import ArrayUtils
from phyfu.utils.log_utils import FuzzingLogger, ReadableMetaInfo, MetaInfo


class Fuzzer:
    def __init__(self, model: Model, loss_func: LossFunction, seed_getter: SeedGetter,
                 array_utils: ArrayUtils, mutator: Mutator, optimizer: Optimizer,
                 logger: FuzzingLogger):
        self.model = model
        self.cfg = model.config
        self.seed_getter = seed_getter
        self.mutator = mutator
        self.loss_func = loss_func
        self.logger = logger
        self.optimizer = optimizer
        self.array_utils = array_utils

    def fuzz(self):
        for test_time in TimingUtils(self.cfg.test_times):
            root_state = self.seed_getter.next_seed()
            seed_action = self.model.rand_action(self.cfg.num_steps)
            try:
                seed_init, seed_final = self.model.two_stage_step(
                    root_state, seed_action, self.cfg.mut_steps)
                mut_init, mut_final, mut_act, mut_dev = self.mutator.mutate(
                    root_state, seed_action)
            except RuntimeError as e:
                print(e)
                continue
            self.logger.set_num_iter(test_time)
            loss_before = self.array_utils.loss_to_float(
                self.loss_func.apply(mut_final, seed_final))
            try:
                min_loss, min_loss_dev, stop_msg = self.optimizer.gradient_descend(
                    mut_dev, mut_final, root_state, seed_final, seed_action, self.logger)
                min_loss_mut_act = self.model.mutate_seed_action(
                    min_loss_dev, seed_action, self.cfg.mut_steps)
                min_loss_init, min_loss_final = self.model.two_stage_step(
                    root_state, min_loss_mut_act, self.cfg.mut_steps
                )
            except RuntimeError as e:
                min_loss, min_loss_dev, stop_msg = loss_before, mut_dev, str(e)
                min_loss_init, min_loss_final = mut_init, mut_final
            meta_info = MetaInfo(root_state, seed_action, seed_init, seed_final,
                                 mut_dev, mut_init, mut_final, min_loss, min_loss_dev,
                                 min_loss_init, min_loss_final)
            readable_meta = ReadableMetaInfo(
                test_time, mut_dev, loss_before, min_loss, min_loss_dev, stop_msg
            )
            self.logger.log_meta_info(meta_info, readable_meta)
            self.logger.dump_test_iter()

        self.logger.dump_summary()
