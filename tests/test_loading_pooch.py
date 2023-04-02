from pathlib import Path

from sleplet._data.setup_pooch import find_on_pooch_then_local


def test_known_file_exists_on_pooch() -> None:
    """Checks that a known file exists on the Pooch repository."""
    filename = "slepian_masks_south_america_L1.npy"
    assert isinstance(find_on_pooch_then_local(filename), str)


def test_known_file_exists_in_data_folder() -> None:
    """Checks that a known file exists in the data folder."""
    filename = "meshes_regions_bird.toml"
    assert isinstance(find_on_pooch_then_local(filename), Path)


def test_unknown_file_returns_none() -> None:
    """Checks made up file doesn't exist."""
    filename = "xyz"
    assert find_on_pooch_then_local(filename) is None
