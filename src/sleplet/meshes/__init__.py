"""
classes to create and handle mesh (manifold) data
"""

from ._mesh import Mesh
from ._mesh_slepian import MeshSlepian

__all__ = [
    "Mesh",
    "MeshSlepian",
]
