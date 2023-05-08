import numpy as np
from phyfu.common.loss_func import LossFunction
from phyfu.taichi_mutate.model_loader import QP


class TaichiLossFunc(LossFunction):

    def linear_loss(self, output: QP, label: QP):
        return np.sum(np.abs(output.x - label.x)) + np.sum(np.abs(output.v - label.v))

    def square_loss(self, output, label):
        return np.sum(np.square(output.x - label.x)) + np.sum(np.square(output.v - label.v))
