import threading
import time

import pytest

from robo_advisor.graph.engine import Graph, GraphError, GraphHalted, Node
from robo_advisor.models import ReviewFinding, ReviewReport


def n(name, req, prov, fn=None, **kw):
    return Node(name, fn or (lambda st, p=prov: {k: 1 for k in p}), tuple(req), tuple(prov), **kw)


def test_contract_validation():
    with pytest.raises(GraphError, match="requires 'y'"):
        Graph("g", ("x",)).add(n("a", ["y"], ["z"])).build()
    with pytest.raises(GraphError, match="produced by both"):
        Graph("g", ("x",)).add(n("a", ["x"], ["z"])).add(n("b", ["x"], ["z"])).build()
    with pytest.raises(GraphError, match="cycle"):
        Graph("g").add(n("a", ["q"], ["p"])).add(n("b", ["p"], ["q"])).build()


def test_waves_parallelism_and_view_isolation():
    seen = []

    def slow(st, key):
        assert set(st) == {"x"}                # only declared keys are visible
        seen.append(threading.current_thread().name)
        time.sleep(0.2)
        return {key: st["x"] + 1}

    g = Graph("g", ("x",))
    for k in "abc":
        g.add(Node(k, lambda st, k=k: slow(st, k + "_out"), ("x",), (k + "_out",)))
    g.add(n("join", ["a_out", "b_out", "c_out"], ["done"]))
    assert g.waves() == [["a", "b", "c"], ["join"]]
    t = time.time()
    r = g.run({"x": 1})
    assert time.time() - t < 0.5                     # 3 x 0.2s ran concurrently
    assert r.state["a_out"] == 2 and len(set(seen)) == 3
    assert all(e.outputs for e in r.trace)
    assert "join" in g.to_mermaid()


def test_node_must_return_declared_keys():
    g = Graph("g", ("x",)).add(Node("a", lambda st: {"wrong": 1}, ("x",), ("y",)))
    with pytest.raises(GraphError, match="expected"):
        g.run({"x": 1})


def _report(ok):
    return ReviewReport("t", [ReviewFinding("R", "§", "BLOCKER", ok, "m")])


def test_gate_remediation_then_halt():
    calls = {"opt": 0}

    def opt(st):
        calls["opt"] += 1
        return {"w": st.get("remediation", {}).get("attempt", 0)}

    g = Graph("g", ("x",))
    g.add(Node("opt", opt, ("x",), ("w",), optional=("remediation",)))
    g.add(Node("gate", lambda st: {"rev": _report(st["w"] >= 1)}, ("w",), ("rev",), kind="gate",
               remediate=("opt",), max_retries=2))
    g.add(n("after", ["rev"], ["z"]))
    r = g.run({"x": 0})
    assert calls["opt"] == 2 and r.state["w"] == 1 and "remediation" not in r.state
    assert [x.ok for x in r.reviews] == [False, True]

    g2 = Graph("g", ("x",))
    g2.add(n("a", ["x"], ["w"]))
    g2.add(Node("gate", lambda st: {"rev": _report(False)}, ("w",), ("rev",), kind="gate"))
    g2.add(n("after", ["rev"], ["z"]))
    with pytest.raises(GraphHalted) as e:
        g2.run({"x": 0})
    assert e.value.node == "gate" and "z" not in e.value.result.state
