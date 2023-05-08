from abc import abstractmethod, ABC

class LossFunction(ABC):
    def __init__(self, loss_name):
        if loss_name == "linear":
            self.loss_func = self.linear_loss
        else:
            self.loss_func = self.square_loss

    @abstractmethod
    def linear_loss(self, output, label):
        ...

    @abstractmethod
    def square_loss(self, output, label):
        ...

    def apply(self, output, label):
        return self.loss_func(output, label)


