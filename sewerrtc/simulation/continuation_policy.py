from __future__ import annotations


CONTINUATION_POLICIES = {
    "fixed_anchor_continuation_after_30min": {
        "free_steps": 3,
        "total_steps": 12,
        "description": "first 30 min planned residual, then fixed selected fallback anchor",
    },
    "one_step_action_advantage": {
        "free_steps": 1,
        "total_steps": 12,
        "description": "evaluate first executed 10 min action with frozen continuation",
    },
}

