from phyfu.array_utils.array_interface import ArrayUtils
from phyfu.common.model_loader import Model


class SeedGetter:
    def __init__(self, au: ArrayUtils, model: Model):
        self.au = au
        self.model = model
        self.state = model.default_state
        self.i = 0
        self.min_steps = max(model.config.seed_getter.min_steps,
                             model.config.mut_steps)
        self.max_steps = model.config.seed_getter.max_steps
        self.reset_freq = model.config.seed_getter.reset_freq
        self.num_steps = 0
        self.n_reset_times = 0

    def handle_error(self):
        self.n_reset_times += 1
        print(f"Cannot find a valid seed state. Restarting for the {self.n_reset_times}th time")
        self.i = 0
        if self.n_reset_times > 30:
            # Throw an error message and quit
            raise RuntimeError(
                f"Cannot find a valid seed state even after restarting for "
                f"{self.n_reset_times} times. Quiting.")
        return self.next_seed()

    def next_seed(self):
        if self.i % self.reset_freq == 0:
            self.state = self.model.default_state
        self.i += 1
        num_steps = self.au.random.randint(self.min_steps, self.max_steps)
        action_list = self.model.rand_action(num_steps)
        self.num_steps = num_steps
        try:
            self.state = self.model.step(self.state, action_list)
        except RuntimeError:
            return self.handle_error()

        if not self.model.is_valid_state(self.state):
            return self.handle_error()

        self.n_reset_times = 0

        return self.state
