import time


class TimingUtils:
    def __init__(self, total_iter):
        self.total_iter = total_iter
        self.cur_iter = 1
        self.start = time.time()

    @staticmethod
    def hr_min_s(sec):
        r = int(sec)
        hr = r // 3600
        r = r - hr * 3600
        mi = r // 60
        r -= mi * 60
        return f"{hr}hr {mi}min {r}sec"

    def __iter__(self):
        return self

    def __next__(self):
        if self.cur_iter > self.total_iter:
            raise StopIteration
        self.step()
        self.cur_iter += 1
        return self.cur_iter - 1
    
    def step(self):
        print("=====================")
        print("test_time:", self.cur_iter)
        if self.cur_iter == 1:
            return
        past_time = time.time() - self.start
        print(f"Executed time: {TimingUtils.hr_min_s(past_time)}")
        print(f"Estimated remaining time: {TimingUtils.hr_min_s(past_time / (self.cur_iter - 1) * (self.total_iter + 1 - self.cur_iter))}")
