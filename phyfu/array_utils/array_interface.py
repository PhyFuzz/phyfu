from abc import ABC, abstractmethod


class ArrayUtils(ABC):
    @staticmethod
    @abstractmethod
    def length(v):
        ...

    @staticmethod
    @abstractmethod
    def angle(v1, v2):
        ...

    @staticmethod
    @abstractmethod
    def to_numpy(arr):
        ...

    @staticmethod
    @abstractmethod
    def zeros(shape):
        ...

    @staticmethod
    @abstractmethod
    def euc_dist(a1, a2):
        ...

    @staticmethod
    @abstractmethod
    def tile(a, rep):
        ...

    @staticmethod
    @abstractmethod
    def concatenate(arrays):
        ...

    @property
    @abstractmethod
    def random(self):
        ...

    @staticmethod
    def flatten(array):
        return array.flatten()

    @staticmethod
    @abstractmethod
    def loss_to_float(loss) -> float:
        """
        Convert the loss to float
        :param loss: the loss in the array format
        :return: a float value
        """

    @staticmethod
    @abstractmethod
    def save(file, arr):
        """
        Save an array to file
        :param file: the path of the file to be written to
        :param arr: the array to be saved
        :return: None
        """

    @staticmethod
    @abstractmethod
    def load(file):
        """
        Load arrays or pickled objects from file
        :param file: the file to be loaded from
        :return: an array
        """
