import math
from collections import deque


class LossUtils:
    def __init__(self, loss_threshold, max_epochs,
                 max_len, display_freq) -> None:
        self.q = deque(maxlen=max_len)
        self.times = 0
        self.display_freq = display_freq
        self.min_loss = float('inf')
        self.min_loss_item = None
        self.terminate_message = None
        self.max_epochs = max_epochs
        self.loss_threshold = loss_threshold

    def get_min_loss_item(self):
        return self.min_loss_item

    def get_min_loss(self):
        return self.min_loss
    
    def add_item(self, loss: float, item):
        if len(self.q) > self.q.maxlen:
            self.q.popleft()
        self.q.append(loss)
        if loss < self.min_loss:
            self.min_loss = loss
            if hasattr(item, 'clone'):
                self.min_loss_item = item.clone()
            elif hasattr(item, 'copy'):
                self.min_loss_item = item.copy()
            else:
                self.min_loss_item = item
        if self.times % self.display_freq == 0:
            print(loss)
        self.times += 1

    def has_converged(self):
        if self.min_loss < self.loss_threshold:
            self.terminate_message = "min loss small enough"
            return True
        if any([math.isnan(i) for i in self.q]):
            self.terminate_message = 'nan'
            return True
        if any([math.isinf(i) for i in self.q]):
            self.terminate_message = 'inf'
            return True
        if any([i > 200 for i in self.q]):
            self.terminate_message = 'Loss greater than 200'
            return True
        if len(self.q) < self.q.maxlen:
            return False
        if self.times >= self.max_epochs:
            self.terminate_message = f"Reached {self.max_epochs} epochs"
            return True
        prev_i = self.q[0]
        inc_min, inc_max = float('inf'), -float('inf')
        rel_inc_max = -float('inf')
        n_occ, n_conv, n_inc = 0, 0, 0
        for t in range(1, len(self.q)):
            i = self.q[t]
            if i > prev_i * 1.1:
                n_occ += 1
            if prev_i >= i and prev_i - i < 0.01 * self.loss_threshold:
                n_conv += 1
            if i > prev_i:
                if prev_i < inc_min:
                    inc_min = prev_i
                if i > inc_max:
                    inc_max = i
                if abs(i / prev_i) > rel_inc_max:
                    rel_inc_max = abs(i / prev_i)
                n_inc += 1
            prev_i = i
        if n_occ >= self.q.maxlen / 2.2:
            self.terminate_message = f"n_occ: {n_occ}"
            return True
        if n_conv > self.q.maxlen / 2:
            self.terminate_message = f"n_conv: {n_conv}"
            return True
        if n_inc > self.q.maxlen / 2:
            self.terminate_message = f"n_inc: {n_inc}"
            return True
        if rel_inc_max > 3:
            self.terminate_message = f"rel_inc_max: {rel_inc_max}"
            return True
        # rel_inc = (inc_max - inc_min) / max(inc_min, threshold)
        # if rel_inc > 1.5:
        #     print("rel_inc:", rel_inc)
        #     return True
        return False
