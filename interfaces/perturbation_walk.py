import abc

from abc import abstractmethod
from .perturbation_chain import PerturbationChain

class PerturbationWalk[T](abc.ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def next(self, chain: PerturbationChain) -> T:
        pass