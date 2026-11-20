import abc

from typing import Any
from abc import abstractmethod

from .graph_types import GraphLike

class PerturbationHeuristic[T](abc.ABC):
    def __init__(self, graph: GraphLike, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def next(self, graph: GraphLike, chain: Any) -> PerturbationHeuristic[T] | None:
        pass

    @abstractmethod
    def current(self) -> T:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def soft_reset(self) -> None:
        pass

    @abstractmethod
    def add(self, element: T) -> None:
        pass