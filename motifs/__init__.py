from interfaces import MotifGenerator

from .cycle_motif_generator import CycleMotifGenerator
from .house_motif_generator import HouseMotifGenerator
from .chain_motif_generator import ChainMotifGenerator
from .star_motif_generator import StarMotifGenerator
from .gate_motif_generator import GateMotifGenerator

__all__ = ['CycleMotifGenerator', 'HouseMotifGenerator', 'ChainMotifGenerator', 'StarMotifGenerator', 'GateMotifGenerator']

motif_generators = {
    "cycle": CycleMotifGenerator,
    "house": HouseMotifGenerator,
    "chain": ChainMotifGenerator,
    "star": StarMotifGenerator,
    "gate": GateMotifGenerator,
}

# Maps motif base names to the keyword argument names expected by
# each generator's generate_motif() method (positional order).
motif_param_names = {
    "cycle": ["len_cycle"],
    "house": [],
    "chain": ["length"],
    "star": ["num_leaves"],
    "gate": ["arm_length"],
}