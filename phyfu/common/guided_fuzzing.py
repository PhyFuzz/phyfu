from phyfu.common.mutate import Mutator
from phyfu.common.art_seed_getter import ArtSeedGetter
from phyfu.common.model_loader import Model
from phyfu.common.bug_oracle import BugOracle
from phyfu.common.loss_func import LossFunction
from phyfu.common.optimizer import Optimizer

from phyfu.utils.timing_utils import TimingUtils
from phyfu.array_utils.array_interface import ArrayUtils
from phyfu.utils.log_utils import FuzzingLogger, ReadableMetaInfo, MetaInfo


class GuidedFuzzer:
    def __init__(self, model: Model, loss_func: LossFunction, seed_getter: ArtSeedGetter,
                 array_utils: ArrayUtils, mutator: Mutator, optimizer: Optimizer,
                 bug_checker: BugOracle, logger: FuzzingLogger):
        self.model = model
        self.cfg = model.config
        self.seed_getter = seed_getter
        self.mutator = mutator
        self.loss_func = loss_func
        self.logger = logger
        self.optimizer = optimizer
        self.array_utils = array_utils
        self.bug_checker = bug_checker

    def fuzz(self):
        for test_time in TimingUtils(self.cfg.test_times):
            for i in range(2):
                root_state, seed_action = self.seed_getter.guided_gen_seed()
                try:
                    seed_init, seed_final = self.model.two_stage_step(
                        root_state, seed_action, self.cfg.mut_steps)
                    mut_init, mut_final, mut_act, mut_dev = self.mutator.mutate(
                        root_state, seed_action)
                except RuntimeError as e:
                    print(e)
                    self.seed_getter.add_executed_item(root_state, True)
                    continue
                loss_before = self.array_utils.loss_to_float(
                    self.loss_func.apply(mut_final, seed_final))
                if loss_before > 3 * self.bug_checker.oracle_cfg.min_loss_threshold:
                    break
            self.logger.set_num_iter(test_time)
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
                self.seed_getter.add_executed_item(root_state, True)

            diff_before, diff_after = self.bug_checker.get_diff(
                mut_init, min_loss_init, seed_init)

            is_deviated = self.bug_checker.is_deviated(diff_before, diff_after)
            if min_loss > self.bug_checker.oracle_cfg.min_loss_threshold or is_deviated:
                self.seed_getter.add_executed_item(root_state, True)
            else:
                self.seed_getter.add_executed_item(root_state, False)

            meta_info = MetaInfo(root_state, seed_action, seed_init, seed_final,
                             mut_dev, mut_init, mut_final, min_loss, min_loss_dev,
                             min_loss_init, min_loss_final)
            readable_meta = ReadableMetaInfo(
                test_time, mut_dev, loss_before, min_loss, min_loss_dev, stop_msg
            )
            self.logger.log_meta_info(meta_info, readable_meta)
            self.logger.dump_test_iter()

        self.logger.dump_summary()
