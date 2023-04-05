from pathlib import Path

import sleplet


def test_known_file_exists_on_pooch() -> None:
    """Checks that a known file exists on the Pooch repository."""
    filename = "slepian_masks_south_america_L1.npy"
    assert isinstance(sleplet._data.setup_pooch.find_on_pooch_then_local(filename), str)


def test_file_exists_in_data_folder_after_initial_run() -> None:
    """Checks that a file exists in the data folder after it has been saved."""
    new_south_america = sleplet.functions.SouthAmerica(15)
    assert isinstance(
        sleplet._data.setup_pooch.find_on_pooch_then_local(
            f"slepian_masks_{new_south_america.name}.npy",
        ),
        Path,
    )


def test_unknown_file_returns_none() -> None:
    """Checks made up file doesn't exist."""
    filename = "xyz"
    assert sleplet._data.setup_pooch.find_on_pooch_then_local(filename) is None
