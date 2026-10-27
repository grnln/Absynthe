import abc
import networkx as nx

from typing import Any
from abc import abstractmethod

class PerturbationWalk[T](abc.ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def next(self, graph: nx.Graph[Any], chain: Any) -> PerturbationWalk[T] | None:
        pass

    @abstractmethod
    def current(self) -> T:
        pass