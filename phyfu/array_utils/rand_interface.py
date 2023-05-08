from abc import abstractmethod, ABC


class RandUtils(ABC):
    @staticmethod
    @abstractmethod
    def randint(low, high, size=None):
        ...

    @staticmethod
    @abstractmethod
    def rand_normal(mean, sigma, size):
        ...
