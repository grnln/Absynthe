from __future__ import annotations

from .applied_perturbation import AppliedPerturbation

class PerturbationChain:
    def __init__(self, perturbations: list[AppliedPerturbation]) -> None:
        self.perturbations = perturbations

    def __getitem__(self, index: int) -> AppliedPerturbation:
        return self.perturbations[index]

    def __len__(self) -> int:
        return len(self.perturbations)

    def __str__(self) -> str:
        return str(self.perturbations)

    def __repr__(self) -> str:
        return str(self.perturbations)

    def __add__(self, other: PerturbationChain) -> PerturbationChain:
        return PerturbationChain(self.perturbations + other.perturbations)

    def append(self, perturbation: AppliedPerturbation) -> None:
        self.perturbations.append(perturbation)

    def remove(self, perturbation: AppliedPerturbation) -> None:
        self.perturbations.remove(perturbation)

    def clear(self) -> None:
        self.perturbations.clear()