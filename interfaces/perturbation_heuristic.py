import abc
import networkx as nx

from typing import Any
from abc import abstractmethod

class PerturbationHeuristic[T](abc.ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def next(self, graph: nx.Graph[Any], chain: Any) -> PerturbationHeuristic[T] | None:
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