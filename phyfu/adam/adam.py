import numpy as np


class Adam:
    def __init__(self, lr=0.001, decay1=0.9, decay2=0.999, eps=1e-7, clip_norm=None):
        self.cache = {}
        self.cur_step = 0

        self.cache = {}
        self.hyperparameters = {"id": "Adam", "lr": lr, "eps": eps, "decay1": decay1,
                                "decay2": decay2, "clip_norm": clip_norm}

    def __str__(self):
        H = self.hyperparameters
        lr, d1, d2 = H["lr"], H["decay1"], H["decay2"]
        eps, cn, sc = H["eps"], H["clip_norm"], H["lr_scheduler"]
        return f"Adam(lr={lr}, decay1={d1}, decay2={d2}, eps={eps}, " \
               f"clip_norm={cn}, lr_scheduler={sc})"

    def update(self, param, param_grad, param_name):
        C = self.cache
        H = self.hyperparameters
        d1, d2 = H["decay1"], H["decay2"]
        eps, clip_norm = H["eps"], H["clip_norm"]
        lr = H['lr']

        if param_name not in C:
            C[param_name] = {
                "t": 0,
                "mean": np.zeros_like(param_grad),
                "var": np.zeros_like(param_grad),
            }

        # scale gradient to avoid explosion
        t = np.inf if clip_norm is None else clip_norm
        if np.linalg.norm(param_grad) > t:
            param_grad = param_grad * t / np.linalg.norm(param_grad)

        t = C[param_name]["t"] + 1
        var = C[param_name]["var"]
        mean = C[param_name]["mean"]

        # update cache
        C[param_name]["t"] = t
        C[param_name]["var"] = d2 * var + (1 - d2) * param_grad ** 2
        C[param_name]["mean"] = d1 * mean + (1 - d1) * param_grad
        self.cache = C

        # calc unbiased moment estimates and Adam update
        v_hat = C[param_name]["var"] / (1 - d2 ** t)
        m_hat = C[param_name]["mean"] / (1 - d1 ** t)
        update = lr * m_hat / (np.sqrt(v_hat) + eps)
        return param - update
