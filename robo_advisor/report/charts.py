"""Dependency-free inline-SVG charts for the client report.

Colors are CSS custom properties (defined in the report template for light and dark mode),
so charts follow the page theme. Every mark carries a <title> for native hover tooltips, and
each chart in the report is paired with a table view of the same numbers.
"""
from __future__ import annotations

import math
from html import escape
from typing import Callable, Sequence

W, H = 720, 300
PAD_L, PAD_R, PAD_T, PAD_B = 64, 16, 16, 36


def nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    if hi <= lo:
        hi = lo + 1
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    start = math.floor(lo / step + 1e-9) * step
    ticks = [round(start, 10)]
    while ticks[-1] < hi - step * 1e-9:
        ticks.append(round(ticks[-1] + step, 10))
    return ticks


def money_short(v: float) -> str:
    a = abs(v)
    if a >= 1e6:
        return f"${v / 1e6:.1f}M"
    if a >= 1e3:
        return f"${v / 1e3:.0f}k"
    return f"${v:.0f}"


def pct(v: float) -> str:
    return f"{v:.0%}"


class _Frame:
    def __init__(self, x0: float, x1: float, y0: float, y1: float, w: int = W, h: int = H):
        self.x0, self.x1, self.y0, self.y1, self.w, self.h = x0, x1, y0, y1, w, h

    def x(self, v: float) -> float:
        return PAD_L + (v - self.x0) / (self.x1 - self.x0 or 1) * (self.w - PAD_L - PAD_R)

    def y(self, v: float) -> float:
        return self.h - PAD_B - (v - self.y0) / (self.y1 - self.y0 or 1) * (self.h - PAD_T - PAD_B)


def _svg(body: str, label: str, w: int = W, h: int = H) -> str:
    return (f'<svg class="chart" viewBox="0 0 {w} {h}" role="img" aria-label="{escape(label)}" '
            f'preserveAspectRatio="xMidYMid meet">{body}</svg>')


def _y_axis(f: _Frame, ticks: list[float], fmt: Callable[[float], str]) -> str:
    out = []
    for t in ticks:
        y = f.y(t)
        out.append(f'<line class="grid" x1="{PAD_L}" x2="{f.w - PAD_R}" y1="{y:.1f}" y2="{y:.1f}"/>'
                   f'<text class="tick" x="{PAD_L - 8}" y="{y + 4:.1f}" text-anchor="end">{escape(fmt(t))}</text>')
    return "".join(out)


def _x_labels(f: _Frame, labels: Sequence[str], every: int) -> str:
    out = []
    for i in range(0, len(labels), max(every, 1)):
        anchor = "end" if f.x(i) > f.w - PAD_R - 30 else "middle"
        out.append(f'<text class="tick" x="{f.x(i):.1f}" y="{f.h - PAD_B + 18}" text-anchor="{anchor}">'
                   f'{escape(labels[i])}</text>')
    return "".join(out)


def _path(xs, ys) -> str:
    return "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))


def line_chart(labels: Sequence[str], series: dict[str, tuple[Sequence[float], str]],
               fmt: Callable[[float], str] = money_short, hline: tuple[float, str] | None = None,
               bands: list[tuple[Sequence[float], Sequence[float], str, str]] | None = None,
               label: str = "line chart", x_every: int | None = None) -> str:
    """series: name -> (values, css color var). bands: (lo, hi, css var, name)."""
    vals = [v for s, _ in series.values() for v in s]
    for lo, hi, _, _ in bands or []:
        vals += list(lo) + list(hi)
    if hline:
        vals.append(hline[0])
    ymin, ymax = min(0.0, min(vals)), max(vals)
    ticks = nice_ticks(ymin, ymax)
    f = _Frame(0, len(labels) - 1, ticks[0], ticks[-1])
    body = [_y_axis(f, ticks, fmt)]
    xs = [f.x(i) for i in range(len(labels))]
    for lo, hi, color, name in bands or []:
        pts = _path(xs, [f.y(v) for v in hi]) + " L" + " L".join(
            f"{x:.1f},{f.y(v):.1f}" for x, v in zip(reversed(xs), reversed(list(lo)))) + " Z"
        body.append(f'<path d="{pts}" fill="var({color})" stroke="none"><title>{escape(name)}</title></path>')
    if hline:
        y = f.y(hline[0])
        body.append(f'<line class="refline" x1="{PAD_L}" x2="{W - PAD_R}" y1="{y:.1f}" y2="{y:.1f}"/>'
                    f'<text class="reflabel" x="{PAD_L + 6}" y="{y - 6:.1f}">{escape(hline[1])}</text>')
    for name, (s, color) in series.items():
        body.append(f'<path d="{_path(xs, [f.y(v) for v in s])}" fill="none" stroke="var({color})" '
                    f'stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
    # hover targets: one column per x with every series value
    step = (xs[1] - xs[0]) if len(xs) > 1 else 10
    for i, x in enumerate(xs):
        tip = labels[i] + "".join(f"\n{n}: {fmt(s[i])}" for n, (s, _) in series.items())
        body.append(f'<rect class="hit" x="{x - step / 2:.1f}" y="{PAD_T}" width="{step:.1f}" '
                    f'height="{H - PAD_T - PAD_B}"><title>{escape(tip)}</title></rect>')
    body.append(_x_labels(f, labels, x_every or max(1, len(labels) // 8)))
    return _svg("".join(body), label)


def bar_chart_h(labels: Sequence[str], values: Sequence[float], fmt: Callable[[float], str] = pct,
                label: str = "bar chart", pos_color: str = "--series-1", neg_color: str = "--series-8",
                notes: Sequence[str] | None = None) -> str:
    n = len(labels)
    row = 28
    h = PAD_T + n * row + 24
    lo, hi = min(0.0, min(values)), max(0.0, max(values))
    ticks = nice_ticks(lo, hi, 4)
    x0, x1 = ticks[0], ticks[-1]
    left = 64

    def x(v):
        return left + (v - x0) / (x1 - x0 or 1) * (W - left - 70)

    body = []
    for t in ticks:
        body.append(f'<line class="grid" x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{PAD_T}" y2="{h - 24}"/>'
                    f'<text class="tick" x="{x(t):.1f}" y="{h - 8}" text-anchor="middle">{escape(fmt(t))}</text>')
    for i, (lab, v) in enumerate(zip(labels, values)):
        y = PAD_T + i * row + 5
        a, b = sorted((x(0), x(v)))
        color = pos_color if v >= 0 else neg_color
        tip = f"{lab}: {fmt(v)}" + (f"\n{notes[i]}" if notes else "")
        body.append(f'<text class="label" x="{left - 8}" y="{y + 13}" text-anchor="end">{escape(lab)}</text>'
                    f'<rect x="{a:.1f}" y="{y}" width="{max(b - a, 1):.1f}" height="18" rx="4" '
                    f'fill="var({color})"><title>{escape(tip)}</title></rect>'
                    f'<text class="value" x="{(b if v >= 0 else a) + (6 if v >= 0 else -6):.1f}" y="{y + 13}" '
                    f'text-anchor="{"start" if v >= 0 else "end"}">{escape(fmt(v))}</text>')
    return _svg("".join(body), label, W, h)


def histogram(values: Sequence[float], target: float | None, bins: int = 40,
              label: str = "distribution", color: str = "--series-1") -> str:
    vals = sorted(values)
    lo, hi = vals[int(0.005 * len(vals))], vals[int(0.995 * len(vals)) - 1]
    if target is not None:
        lo, hi = min(lo, target * 0.9), max(hi, target * 1.1)
    width = (hi - lo) / bins
    counts = [0] * bins
    for v in vals:
        k = min(bins - 1, max(0, int((v - lo) / width)))
        counts[k] += 1
    n = len(vals)
    shares = [c / n for c in counts]
    ticks = nice_ticks(0, max(shares) * 1.15, 4)
    f = _Frame(lo, hi, 0, ticks[-1])
    body = [_y_axis(f, ticks, lambda v: f"{v:.0%}")]
    for k, s in enumerate(shares):
        a, b = lo + k * width, lo + (k + 1) * width
        xa, xb = f.x(a) + 1, f.x(b) - 1
        above = target is not None and a >= target
        fill = color if (target is None or above) else "--seq-250"
        body.append(f'<rect x="{xa:.1f}" y="{f.y(s):.1f}" width="{max(xb - xa, 1):.1f}" '
                    f'height="{f.y(0) - f.y(s):.1f}" fill="var({fill})"><title>{money_short(a)}-{money_short(b)}: '
                    f'{s:.1%} of paths</title></rect>')
    if target is not None:
        xt = f.x(target)
        body.append(f'<line class="refline" x1="{xt:.1f}" x2="{xt:.1f}" y1="{PAD_T}" y2="{H - PAD_B}"/>'
                    f'<text class="reflabel" x="{xt - 6:.1f}" y="{PAD_T + 12}" text-anchor="end">Target {money_short(target)}</text>')
    for t in nice_ticks(lo, hi, 6):
        if lo <= t <= hi:
            body.append(f'<text class="tick" x="{f.x(t):.1f}" y="{H - PAD_B + 18}" text-anchor="middle">'
                        f'{money_short(t)}</text>')
    return _svg("".join(body), label)


def score_bars(items: list[tuple[str, float]], bands: list[tuple[float, str]], label: str = "risk scores") -> str:
    """Horizontal 0-100 gauges with band boundaries."""
    row, left, h = 34, 150, PAD_T + len(items) * 34 + 26

    def x(v):
        return left + v / 100 * (W - left - 60)

    body = []
    prev = 0.0
    for mx, name in bands:
        body.append(f'<line class="grid" x1="{x(mx):.1f}" x2="{x(mx):.1f}" y1="{PAD_T}" y2="{h - 24}"/>'
                    f'<text class="tick" x="{(x(prev) + x(mx)) / 2:.1f}" y="{h - 8}" text-anchor="middle">'
                    f'{escape(name)}</text>')
        prev = mx
    for i, (name, v) in enumerate(items):
        y = PAD_T + i * row + 6
        color = "--series-1" if i < 2 else "--series-3"
        body.append(f'<text class="label" x="{left - 10}" y="{y + 14}" text-anchor="end">{escape(name)}</text>'
                    f'<rect x="{x(0):.1f}" y="{y}" width="{x(100) - x(0):.1f}" height="20" rx="4" fill="var(--track)"/>'
                    f'<rect x="{x(0):.1f}" y="{y}" width="{max(x(v) - x(0), 2):.1f}" height="20" rx="4" '
                    f'fill="var({color})"><title>{escape(name)}: {v:.1f}</title></rect>'
                    f'<text class="value" x="{x(v) + 6:.1f}" y="{y + 14}">{v:.0f}</text>')
    return _svg("".join(body), label, W, h)
