from pys2sleplet.utils.slepian_mesh_methods import compute_shannon


def test_shannon_less_than_basis_functions(mesh) -> None:
    """
    Shannon number should be less than the total number of basis functions
    """
    shannon = compute_shannon(mesh)
    assert shannon < mesh.basis_functions.shape[0]
