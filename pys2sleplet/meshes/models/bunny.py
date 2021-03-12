from abc import abstractmethod
from dataclasses import dataclass

from pys2sleplet.meshes.mesh import Mesh


@dataclass  # type: ignore
class Bunny(Mesh):
    def __post_init__(self) -> None:
        super().__post_init__()

    @abstractmethod
    def _create_name(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _setup_region(self) -> None:
        raise NotImplementedError
