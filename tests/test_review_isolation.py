import ast
from pathlib import Path

REVIEW = Path(__file__).resolve().parents[1] / "robo_advisor" / "review"
ALLOWED = {"models", "config", "universe", "review", "independent", "reviewer"}
FORBIDDEN = {"estimation", "optimization", "simulation", "benchmark", "projection", "tax", "questionnaire",
             "explain", "rebalancing", "data", "agents"}


def test_reviewer_does_not_import_production_computations():
    for f in REVIEW.glob("*.py"):
        tree = ast.parse(f.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level > 0:
                parts = set((node.module or "").split("."))
                names = {a.name for a in node.names} if not node.module else set()
                bad = (parts | names) & FORBIDDEN
                assert not bad, f"{f.name} imports production module(s) {bad}"
            if isinstance(node, ast.ImportFrom) and node.level == 0 and (node.module or "").startswith("robo_advisor"):
                assert set(node.module.split(".")[1:2]) <= ALLOWED, f"{f.name} imports {node.module}"
