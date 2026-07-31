"""V4.2 pipeline dependencies — PREREQUISITES, no cycles, ALL_STAGES, no-Torch."""
from __future__ import annotations

import sys

import pytest


class TestPrerequisitesCompleteness:
    def test_all_v42_stages_have_prerequisites(self):
        from sewerrtc.v4.pipeline import ALL_STAGES, PREREQUISITES
        v42_stages = [s for s in ALL_STAGES if "V42" in s or "v42" in s.lower()]
        for stage in v42_stages:
            assert stage in PREREQUISITES, (
                f"V4.2 stage '{stage}' missing from PREREQUISITES"
            )

    def test_no_cyclic_dependencies(self):
        from sewerrtc.v4.pipeline import PREREQUISITES
        visited: set[str] = set()
        in_stack: set[str] = set()

        def _dfs(node: str) -> bool:
            if node in in_stack:
                return True  # cycle
            if node in visited:
                return False
            in_stack.add(node)
            visited.add(node)
            for dep in PREREQUISITES.get(node, ()):
                if _dfs(dep):
                    return True
            in_stack.discard(node)
            return False

        for stage in PREREQUISITES:
            if stage not in visited:
                assert not _dfs(stage), f"Cyclic dependency detected involving '{stage}'"


class TestAllStagesContainsNew:
    def test_all_v42_data_pipeline_stages(self):
        from sewerrtc.v4.pipeline import ALL_STAGES
        required = {
            "BuildV42EventUsageLedger",
            "AuditV42EventUsageLedger",
            "BuildV42UnifiedDevelopmentPool",
            "AuditV42UnifiedDevelopmentPool",
            "BuildV42DerivedSupervision",
            "AuditV42DerivedSupervision",
            "PlanV42NestedGroupedCV",
            "AuditV42NestedGroupedCVPlan",
            "RunV42NestedGroupedCV",
            "BuildV42NestedGroupedCVResults",
            "AuditV42NestedGroupedCVResults",
        }
        assert required.issubset(set(ALL_STAGES))

    def test_v42_validation_stages(self):
        from sewerrtc.v4.pipeline import ALL_STAGES
        required = {
            "AuditV42HeadActivation",
            "AuditV42TargetMetricSemantics",
            "AuditV42RankingPhysics",
            "RunV42TinyOverfit",
        }
        assert required.issubset(set(ALL_STAGES))

    def test_v42_fresh_eval_stages(self):
        from sewerrtc.v4.pipeline import ALL_STAGES
        required = {
            "PlanV42FreshEvaluationSplit",
            "AuditV42FreshEvaluationAvailability",
        }
        assert required.issubset(set(ALL_STAGES))


class TestTorchLightImport:
    def test_pipeline_import_does_not_load_torch(self):
        """Verify pipeline imports don't pull in torch (subprocess-isolated)."""
        import subprocess

        result = subprocess.run(
            [sys.executable, "-c", (
                "import sys; "
                "assert 'torch' not in sys.modules; "
                "import importlib; "
                "importlib.import_module('sewerrtc.v4.pipeline'); "
                "importlib.import_module('sewerrtc.v4.pipeline_v42'); "
                "assert 'torch' not in sys.modules, 'torch loaded by pipeline import'"
            )],
            capture_output=True, text=True, timeout=60
        )
        assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
