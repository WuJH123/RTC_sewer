from __future__ import annotations

import pandas as pd

from sewerrtc.v4.v42_r0_paper_dataset_strict import _bool_series
from sewerrtc.v4.v42_reusable_pool_strict import _scalar_bool


def test_string_false_is_not_truthy_in_formal_step2_admission():
    frame = pd.DataFrame({"eligible_formal_all_target": ["False", "True", "0", "1"]})
    assert _bool_series(frame, "eligible_formal_all_target").tolist() == [False, True, False, True]


def test_scalar_false_strings_do_not_pass_four_branch_finite_gate():
    assert _scalar_bool("False", name="finite") is False
    assert _scalar_bool("0", name="finite") is False
    assert _scalar_bool("True", name="finite") is True
    assert _scalar_bool("1", name="finite") is True
