import pandas as pd

from sewerrtc.v4.v42_reusable_pool_strict import _bool


def test_strict_alignment_accepts_merged_forcing_column():
    cases = pd.DataFrame(
        {
            "same_state_numeric_pass": [True],
            "same_forcing_pass_x": [False],
            "same_forcing_pass_y": [True],
        }
    )
    forcing = _bool(cases, "same_forcing_pass_y")
    assert bool((_bool(cases, "same_state_numeric_pass") & forcing).iloc[0])
