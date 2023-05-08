import numpy as np
from dataclasses import dataclass
from typing import Any, List

from phyfu.array_utils.array_interface import ArrayUtils
from phyfu.common.model_loader import Model
from phyfu.common.seed_getter import SeedGetter


@dataclass
class Candidate:
    state: np.ndarray
    act: np.ndarray
    min_dist: float

@dataclass
class ExecutedItem:
    state: Any
    fail: bool


class ArtSeedGetter(SeedGetter):
    def __init__(self, au: ArrayUtils, model: Model):
        super().__init__(au, model)
        self.candidates: List[Candidate] = []
        self.executed_items: List[ExecutedItem] = []
        self.cfg = model.config.seed_getter.art_params
        # self._default_cnt = self.cfg.mut_cnt

    def generate_points(self, n):
        states, act = [], []
        for i in range(n):
            states.append(super().next_seed())
            act.append(self.model.rand_action(self.model.config.num_steps))
        return states, act

    def compute_dist(self, s1, s2) -> float:
        return self.au.euc_dist(self.model.state_embedding(s1),
                                self.model.state_embedding(s2))

    def add_executed_item(self, state, fail: bool):
        if not fail and len(self.executed_items) < 1000:
            if len(self.executed_items) < 10 or np.random.random() < 0.1:
                self.executed_items.append(ExecutedItem(state, fail))
                for cand in self.candidates:
                    cand.min_dist = min(cand.min_dist, self.compute_dist(cand.state, state))

    def guided_gen_seed(self):
        if len(self.executed_items) < self.cfg.init_pop_size:
            states, act = self.generate_points(1)
            return states[0], act[0]

        cand_states = [c.state for c in self.candidates]
        cand_act = [c.act for c in self.candidates]
        min_dist_list = [c.min_dist for c in self.candidates]

        if len(cand_states) < self.cfg.cand_size:
            new_states, new_act = self.generate_points(
                self.cfg.cand_size - len(cand_states))
            for s, a in zip(new_states, new_act):
                min_dist = min(self.compute_dist(s, executed.state)
                               for executed in self.executed_items)
                cand_states.append(s)
                cand_act.append(a)
                min_dist_list.append(min_dist)

        np_min_dist_list = np.array(min_dist_list)
        p = np_min_dist_list / np.sum(np_min_dist_list)
        sel_cand_idx = np.random.choice(len(p), p=p)

        ids_to_remove = [sel_cand_idx]

        if np.random.random() < self.cfg.refresh_prob and len(self.candidates) > 10:
            n_to_remove = len(self.candidates) // 10
            idx_list = np.argpartition(np_min_dist_list, n_to_remove)[:n_to_remove]
            ids_to_remove.extend(idx_list.tolist())

        self.candidates = [Candidate(cand_states[i], cand_act[i], min_dist_list[i])
                           for i in range(len(cand_states)) if i not in ids_to_remove]

        return cand_states[sel_cand_idx], cand_act[sel_cand_idx]
