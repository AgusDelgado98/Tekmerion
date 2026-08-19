"""ML layer stays out of Evidence and the production pipeline."""

from __future__ import annotations

import ast
from pathlib import Path


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def test_ml_package_does_not_import_evidence_or_app():
    root = Path("analysis/ml")
    blocked = ("analysis.evidence", "app")
    for py in root.glob("*.py"):
        imported = _imports(py)
        for name in imported:
            for prefix in blocked:
                assert not name.startswith(prefix), f"{py} imports {name}"


def test_label_and_harvest_do_not_import_classifiers():
    blocked = "analysis.classifiers"
    for rel in (
        "analysis/ml/label.py",
        "analysis/ml/harvest.py",
        "analysis/ml/gold.py",
        "analysis/ml/fetch_candidates.py",
        "analysis/ml/gate.py",
    ):
        imported = _imports(Path(rel))
        for name in imported:
            assert not name.startswith(blocked), f"{rel} imports {name}"


def test_evidence_does_not_import_ml():
    imported = _imports(Path("analysis/evidence.py"))
    for name in imported:
        assert not name.startswith("analysis.ml"), name
