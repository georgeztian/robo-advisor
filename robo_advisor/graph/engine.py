"""A small typed DAG engine for agent workflows ("graph engineering").

* Nodes declare the blackboard keys they ``requires`` / ``provides`` (plus ``optional`` keys).
  Edges are derived from those contracts; the graph is validated at build time (acyclic,
  every requirement has exactly one producer or is a declared input).
* Execution proceeds in topological generations ("waves"); independent nodes in a wave run
  concurrently on a thread pool.
* A node only sees its declared keys (read-only view) and must return exactly the keys it
  provides; outputs are content-hashed into a provenance trace.
* Gate nodes return a ReviewReport. A report with BLOCKER findings either triggers the gate's
  remediation (re-running the named upstream nodes and everything between them and the
  gate, with a ``remediation`` hint on the blackboard) or halts the run with GraphHalted.
"""
from __future__ import annotations

import hashlib
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Literal, Mapping

import networkx as nx

from ..models import ReviewReport

NodeFn = Callable[[Mapping[str, Any]], dict[str, Any]]


class GraphError(RuntimeError):
    pass


class GraphHalted(RuntimeError):
    def __init__(self, message: str, node: str, report: ReviewReport | None, result: "RunResult"):
        super().__init__(message)
        self.node, self.report, self.result = node, report, result


@dataclass
class Node:
    name: str
    fn: NodeFn
    requires: tuple[str, ...]
    provides: tuple[str, ...]
    optional: tuple[str, ...] = ()
    kind: Literal["agent", "gate"] = "agent"
    description: str = ""
    remediate: tuple[str, ...] = ()
    max_retries: int = 0


@dataclass
class TraceEvent:
    node: str
    kind: str
    wave: int
    attempt: int
    started: float
    seconds: float
    status: str
    outputs: dict[str, str] = field(default_factory=dict)   # key -> sha256[:12]
    detail: str = ""


@dataclass
class RunResult:
    state: dict[str, Any]
    trace: list[TraceEvent]
    reviews: list[ReviewReport]

    def timeline(self) -> str:
        return "\n".join(f"w{e.wave} {e.node:<18} {e.kind:<5} try{e.attempt} {e.status:<8} "
                         f"{e.seconds * 1000:8.1f} ms {e.detail}" for e in self.trace)


def _digest(v: Any) -> str:
    try:
        return hashlib.sha256(pickle.dumps(v, protocol=5)).hexdigest()[:12]
    except Exception:  # pragma: no cover - unpicklable values still get a stable-ish id
        return hashlib.sha256(repr(v).encode()).hexdigest()[:12]


class Graph:
    def __init__(self, name: str, inputs: tuple[str, ...] = ()):
        self.name = name
        self.inputs = tuple(inputs)
        self.nodes: dict[str, Node] = {}
        self._g: nx.DiGraph | None = None

    def add(self, node: Node) -> "Graph":
        if node.name in self.nodes:
            raise GraphError(f"duplicate node {node.name}")
        self.nodes[node.name] = node
        self._g = None
        return self

    # ------------------------------------------------------------------ build / validate
    def build(self) -> nx.DiGraph:
        if self._g is not None:
            return self._g
        producer: dict[str, str] = {k: "__input__" for k in self.inputs}
        for n in self.nodes.values():
            for k in n.provides:
                if k in producer:
                    raise GraphError(f"key {k!r} produced by both {producer[k]} and {n.name}")
                producer[k] = n.name
        g = nx.DiGraph(name=self.name)
        for n in self.nodes.values():
            g.add_node(n.name, kind=n.kind)
        for n in self.nodes.values():
            for k in n.requires:
                if k not in producer:
                    raise GraphError(f"node {n.name} requires {k!r} but nothing provides it")
                if producer[k] != "__input__":
                    g.add_edge(producer[k], n.name, key=k)
            for k in n.optional:
                if k in producer and producer[k] not in ("__input__", n.name):
                    g.add_edge(producer[k], n.name, key=k)
        if not nx.is_directed_acyclic_graph(g):
            raise GraphError(f"graph {self.name} has a cycle: {nx.find_cycle(g)}")
        for n in self.nodes.values():
            for r in n.remediate:
                if r not in self.nodes or not nx.has_path(g, r, n.name):
                    raise GraphError(f"gate {n.name} remediates {r}, which is not upstream of it")
        self._g = g
        return g

    def waves(self) -> list[list[str]]:
        return [sorted(w) for w in nx.topological_generations(self.build())]

    def to_mermaid(self) -> str:
        g = self.build()
        lines = ["flowchart TD"]
        for name, n in self.nodes.items():
            lines.append(f"  {name}{{{{{name}}}}}" if n.kind == "gate" else f"  {name}[{name}]")
        for a, b, d in g.edges(data=True):
            lines.append(f"  {a} -->|{d['key']}| {b}")
        for name, n in self.nodes.items():
            for r in n.remediate:
                lines.append(f"  {name} -. remediate .-> {r}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ execution
    def _exec(self, node: Node, state: dict, wave: int, attempt: int, trace: list) -> dict:
        keys = node.requires + tuple(k for k in node.optional if k in state)
        view = MappingProxyType({k: state[k] for k in keys})
        t0 = time.time()
        try:
            out = node.fn(view)
        except Exception as e:
            trace.append(TraceEvent(node.name, node.kind, wave, attempt, t0, time.time() - t0,
                                    "error", detail=f"{type(e).__name__}: {e}"))
            raise
        if not isinstance(out, dict) or set(out) != set(node.provides):
            raise GraphError(f"node {node.name} returned keys {sorted(out) if isinstance(out, dict) else out}, "
                             f"expected {sorted(node.provides)}")
        status, detail = "ok", ""
        if node.kind == "gate":
            rep = next(v for v in out.values() if isinstance(v, ReviewReport))
            status = "pass" if rep.ok else "blocked"
            detail = rep.summary()
        trace.append(TraceEvent(node.name, node.kind, wave, attempt, t0, time.time() - t0, status,
                                {k: _digest(v) for k, v in out.items()}, detail))
        return out

    def run(self, inputs: dict[str, Any], parallel: bool = True, max_workers: int = 4) -> RunResult:
        missing = set(self.inputs) - set(inputs)
        if missing:
            raise GraphError(f"missing graph inputs {sorted(missing)}")
        g = self.build()
        state: dict[str, Any] = dict(inputs)
        trace: list[TraceEvent] = []
        reviews: list[ReviewReport] = []
        result = RunResult(state, trace, reviews)
        order = {n: i for i, n in enumerate(nx.topological_sort(g))}
        pool = ThreadPoolExecutor(max_workers=max_workers) if parallel else None
        try:
            for wi, wave in enumerate(self.waves()):
                agents = [self.nodes[n] for n in wave if self.nodes[n].kind == "agent"]
                gates = [self.nodes[n] for n in wave if self.nodes[n].kind == "gate"]
                if pool and len(agents) > 1:
                    futs = [pool.submit(self._exec, n, state, wi, 0, trace) for n in agents]
                    outs = [f.result() for f in futs]
                else:
                    outs = [self._exec(n, state, wi, 0, trace) for n in agents]
                for o in outs:
                    state.update(o)
                for gate in gates:
                    self._run_gate(gate, state, wi, trace, reviews, order, result)
        finally:
            if pool:
                pool.shutdown(wait=True)
        return result

    def _run_gate(self, gate: Node, state: dict, wi: int, trace: list, reviews: list,
                  order: dict[str, int], result: RunResult) -> None:
        g = self.build()
        attempt = 0
        while True:
            out = self._exec(gate, state, wi, attempt, trace)
            state.update(out)
            rep = next(v for v in out.values() if isinstance(v, ReviewReport))
            reviews.append(rep)
            if rep.ok:
                state.pop("remediation", None)
                return
            if attempt >= gate.max_retries or not gate.remediate:
                raise GraphHalted(f"{gate.name}: " + "; ".join(f.message for f in rep.blocking),
                                  gate.name, rep, result)
            attempt += 1
            state["remediation"] = {"gate": gate.name, "attempt": attempt,
                                    "findings": [f.rule_id for f in rep.blocking]}
            # re-run the remediation targets and every node between them and the gate
            rerun = set()
            for r in gate.remediate:
                rerun |= ({r} | nx.descendants(g, r)) & nx.ancestors(g, gate.name)
            for name in sorted(rerun, key=order.get):
                state.update(self._exec(self.nodes[name], state, wi, attempt, trace))
