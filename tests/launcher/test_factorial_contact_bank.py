"""The final40K publication keeps precomputed command semantics fail-closed."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from prepare_factorial_contact_bank import validate_commands


def test_precomputed_commands_accept_decoupled_turn_and_forward():
    command = np.asarray([[0, 0, 0], [0.15, 0, 0], [0, 0, -0.2]], dtype=np.float32)
    validate_commands(command, np.asarray([0, 1, 2], dtype=np.uint8))


@pytest.mark.parametrize("case", ["lateral", "overlap", "phase", "nan", "float_phase", "length", "shape"])
def test_precomputed_commands_reject_bad_inputs(case):
    command = np.asarray([[0, 0, 0], [0.15, 0, 0], [0, 0, -0.2]], dtype=np.float32)
    phase = np.asarray([0, 1, 2], dtype=np.uint8)
    if case == "lateral":
        command[1, 1] = 0.1
    elif case == "overlap":
        command[1, 2] = 0.1
    elif case == "phase":
        phase[1] = 0
    elif case == "nan":
        command[1, 0] = np.nan
    elif case == "float_phase":
        phase = phase.astype(np.float32)
    elif case == "length":
        phase = phase[:-1]
    else:
        command = command[:, :2]
    with pytest.raises(ValueError):
        validate_commands(command, phase)
