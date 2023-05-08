import torch

from phyfu.common.loss_func import LossFunction


class NimbleLossFunc(LossFunction):
    def linear_loss(self, output: torch.Tensor, label: torch.Tensor):
        return torch.sum(torch.abs(output - label))

    def square_loss(self, output, label):
        return torch.sum(torch.square(output - label))