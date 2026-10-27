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
        return f'Perturbation chain of length {len(self)} [\n{",\n".join([f"\t{perturbation}" for perturbation in self.perturbations])}\n]'

    def __repr__(self) -> str:
        return str(self)

    def __add__(self, other: PerturbationChain) -> PerturbationChain:
        return PerturbationChain(self.perturbations + other.perturbations)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PerturbationChain):
            return False
        
        if len(other) != len(self):
            return False

        num_equal_elements = 0

        for perturbation in self:
            for other_perturbation in other:
                if perturbation == other_perturbation:
                    num_equal_elements += 1

        return num_equal_elements == len(self)

    def append(self, perturbation: AppliedPerturbation) -> None:
        self.perturbations.append(perturbation)

    def remove(self, perturbation: AppliedPerturbation) -> None:
        self.perturbations.remove(perturbation)

    def clear(self) -> None:
        self.perturbations.clear()