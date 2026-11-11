"""Turn the saved training runs into figures and tables.

    python report.py                 # reads results/runs, writes results/figures and RESULTS.md
    python report.py --out trial     # the same for another results folder

Nothing is trained here. Every number comes from the files run.py and bench.py wrote.
"""
import argparse
import base64
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from netsize import experiments as ex
from netsize.nets import first_layer_share
from netsize.store import load_records
from netsize.tasks import MAIN_TASKS, N_BIG, get_task
from netsize.train import GRID, GRID_SPAN

# ---------------------------------------------------------------------------
# Look
# ---------------------------------------------------------------------------

SURFACE, INK, INK2, MUTED, GRIDLINE, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#a9a8a1"
RAMP5 = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]          # one hue, light to dark: ordered things
LIGHT_BLUE, DARK_BLUE = "#86b6ef", "#1c5cab"
ACT_COLOR = {"relu": "#2a78d6", "leaky_relu": "#eda100", "gelu": "#008300", "tanh": "#e87ba4"}
ACT_MARK = {"relu": "o", "leaky_relu": "s", "gelu": "^", "tanh": "D"}
ACT_NAME = {"relu": "ReLU", "leaky_relu": "Leaky ReLU", "gelu": "GELU", "tanh": "tanh"}
SEQ = LinearSegmentedColormap.from_list("seq", ["#f3f7fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#6da7ec", "#f0efec", "#ee8a89", "#c93534"])
INKMAP = LinearSegmentedColormap.from_list("ink", [SURFACE, "#52514e", "#0b0b0b"])

TITLE = {"circle": "Circle", "spirals": "Spirals", "digits96": "Small digits", "needle": "Needles",
         "sensors": "Sensors", "digits1000": "Big digits"}
LINE_DEPTHS = (1, 2, 3, 8, 12)


def style():
    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRIDLINE, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 8,
        "axes.labelcolor": INK2, "axes.labelsize": 9, "text.color": INK,
        "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "xtick.major.size": 3, "ytick.major.size": 0,
        "xtick.minor.size": 0, "lines.linewidth": 2, "lines.markersize": 6,
        "legend.frameon": False, "legend.fontsize": 8.5, "savefig.dpi": 170, "figure.dpi": 100,
    })


def short(n):
    """30, 1k, 30k, 1.2M"""
    if n is None:
        return "-"
    if n >= 1e6:
        return f"{n / 1e6:.3g}M"
    if n >= 1000:
        return f"{n / 1000:.3g}k"
    return f"{n:.0f}"


def pct(a, digits=1):
    return f"{100 * a:.{digits}f}"


def title_of(task):
    t = get_task(task)
    return f"{TITLE.get(task, t.title)}  ·  {t.n_in:,} inputs"


def log_params_axis(ax, lo, hi):
    ax.set_xscale("log")
    ticks = [t for t in (10, 100, 1000, 10000, 100000, 1000000) if lo * 0.5 <= t <= hi * 2]
    ax.set_xticks(ticks)
    ax.set_xticklabels([short(t) for t in ticks])
    ax.set_xlim(lo * 0.7, hi * 1.45)
    ax.minorticks_off()


def acc_axis(ax, chance, top=100.0):
    lo = 5 * np.floor(100 * chance / 5) - 5
    ax.set_ylim(lo, top + (top - lo) * 0.04)


def marker(color, shape="o"):
    return dict(marker=shape, color=color, markeredgecolor=SURFACE, markeredgewidth=1.1)


def panels(n=6, cols=3, w=9.6, h=2.75, **kw):
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(w, h * rows), squeeze=False, **kw)
    return fig, list(axes.reshape(-1))


def save(fig, out, name, rect=None):
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)
    fig.tight_layout(rect=rect)
    fig.savefig(os.path.join(out, "figures", name))
    plt.close(fig)
    print("  wrote", name)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load(out):
    """Every trained network, summarised: key -> dictionary.

    A network's accuracy is the middle one of its three seeds (the median), with
    the lowest and highest kept beside it. The middle is used rather than the
    average because a training run occasionally dies outright (see "collapsed"
    below), and one dead run out of three would otherwise drag a good network's
    number down to something no run actually scored.
    """
    nets = {}
    for key, rs in load_records(out).items():
        chosen = sorted((r for r in rs if r["chosen"]), key=lambda r: r["seed"])
        first = chosen[0]
        t = get_task(first["task"])
        test = [r["test_acc"] for r in chosen]
        best_any = max(r["test_acc"] for r in rs)
        net = dict(
            key=key, task=first["task"], widths=tuple(first["widths"]), activation=first["activation"],
            params=first["params"], weights=first["weights"], lr=first["lr"], steps=first["steps"],
            test=test, acc=float(np.median(test)), mean=float(np.mean(test)), lo=min(test), hi=max(test),
            train=float(np.median([r["train_acc"] for r in chosen])),
            dead=float(np.mean([r["dead"] for r in chosen])), curve=first["curve"],
            search=sorted((r["lr"], r["val_acc"], r["test_acc"]) for r in rs if r["role"] == "search"),
            runs=len(rs),
            # a run "collapsed" if it ended at chance although the same network learned something in another run
            collapsed=[(r["lr"], r["seed"], r["chosen"]) for r in rs
                       if r["test_acc"] < t.chance + 0.03 and best_any > t.chance + 0.15],
        )
        if "map" in first:
            net["map"] = np.frombuffer(base64.b64decode(first["map"]), dtype=np.uint8).reshape(GRID, GRID) / 255.0
        if "filters" in first:
            net["filters"] = np.frombuffer(base64.b64decode(first["filters"]), dtype=np.int8).reshape(first["filters_shape"]) / 127.0
        nets[key] = net
    return nets


def collect(entries, nets):
    """Experiment entries joined to their results (entries that were not trained are dropped)."""
    out = []
    for e in entries:
        if e["key"] in nets:
            out.append({**e, **nets[e["key"]]})
    return out


def load_cost(out):
    path = os.path.join(out, "cost.csv")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["widths"] = tuple(int(w) for w in r["widths"].split("-")) if r["widths"] else ()
        for k in ("n_in", "n_out", "params", "depth"):
            r[k] = int(r[k])
        for k in ("train_ms", "infer_us"):
            r[k] = float(r[k])
    return rows


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_tasks(out):
    """What each task looks like."""
    fig = plt.figure(figsize=(9.6, 6.0))
    grid = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.16)
    rng = np.random.default_rng(0)

    def scatter(ax, task):
        x, y = get_task(task).fixed("test")
        x, y = x.numpy()[:1600], y.numpy()[:1600]
        for cls, color in ((0, BLUE), (1, ORANGE)):
            ax.scatter(x[y == cls, 0], x[y == cls, 1], s=5, color=color, linewidths=0)
        ax.set_aspect("equal")
        ax.set_ylim(-GRID_SPAN, GRID_SPAN)

    def mosaic(task, width, height, rows=2, cols=5):
        x, y = get_task(task).fixed("test")
        x, y = x.numpy(), y.numpy()
        gap = max(1, width // 8)
        sheet = np.full((rows * height + (rows - 1) * gap, cols * width + (cols - 1) * gap), np.nan)
        for d in range(rows * cols):
            img = x[np.flatnonzero(y == d)[0]].reshape(height, width)
            r, c = divmod(d, cols)
            sheet[r * (height + gap):r * (height + gap) + height, c * (width + gap):c * (width + gap) + width] = img
        return sheet

    specs = [("circle", "Is the point inside the circle?"), ("spirals", "Which of the two spiral arms is the point on?"),
             ("digits96", "Which digit? 8 x 12 pixels, 24 possible places"),
             ("needle", "10 of the 1,000 inputs matter (drawn 25 x 40)"),
             ("sensors", "One spiral point as its 1,000 sensor readings"),
             ("digits1000", "Which digit? 25 x 40 pixels, two sizes, any place")]
    for i, (task, caption) in enumerate(specs):
        ax = fig.add_subplot(grid[i // 3, i % 3])
        ax.set_title(title_of(task))
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        if task in ("circle", "spirals"):
            scatter(ax, task)
        elif task == "digits96":
            ax.imshow(mosaic(task, 8, 12), cmap=INKMAP, vmin=-0.8, vmax=2.6, interpolation="nearest")
        elif task == "digits1000":
            ax.imshow(mosaic(task, 25, 40), cmap=INKMAP, vmin=-1.2, vmax=2.6, interpolation="nearest")
        elif task == "needle":
            from netsize.tasks import _rng
            where = _rng("needle", "structure").permutation(N_BIG)[:10]
            cells = np.zeros(N_BIG)
            cells[where[:5]] = 1
            cells[where[5:]] = 2
            cmap = matplotlib.colors.ListedColormap(["#ecebe6", BLUE, ORANGE])
            ax.imshow(cells.reshape(25, 40), cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
            ax.set_xticks(np.arange(-0.5, 40), minor=True)
            ax.set_yticks(np.arange(-0.5, 25), minor=True)
            ax.grid(which="minor", color=SURFACE, linewidth=0.6)
            ax.tick_params(which="minor", length=0)
        else:
            x, _ = get_task(task).fixed("test")
            ax.imshow(x.numpy()[int(rng.integers(0, 100))].reshape(25, 40), cmap=DIV, vmin=-0.12, vmax=0.12, interpolation="nearest")
        ax.set_xlabel(caption, color=INK2, fontsize=8.5)
        ax.set_box_aspect(0.74)
        ax.set_adjustable("datalim")
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)
    fig.savefig(os.path.join(out, "figures", "tasks.png"), bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("  wrote tasks.png")


def fig_maps(out, nets, task="spirals", name="maps.png", depths=(1, 2, 3, 8), budgets=(100, 300, 1000, 3000, 10000)):
    """The answer a network gives for every point in the square, at different sizes and depths."""
    got = {(e["depth"], e["budget"]): e for e in collect(ex.grid((task,)), nets) if "map" in e}
    depths = [d for d in depths if any((d, b) in got for b in budgets)]
    if not depths:
        return
    fig, axes = plt.subplots(len(depths), len(budgets), figsize=(9.6, 1.95 * len(depths) + 0.5), squeeze=False)
    for i, d in enumerate(depths):
        for j, b in enumerate(budgets):
            ax = axes[i][j]
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            e = got.get((d, b))
            if e is None:
                ax.text(0.5, 0.5, "not run", ha="center", va="center", color=MUTED, transform=ax.transAxes)
                continue
            ax.imshow(e["map"], cmap=DIV, vmin=0, vmax=1, origin="lower", extent=[-GRID_SPAN, GRID_SPAN] * 2,
                      interpolation="bilinear")
            ax.set_title(f"{pct(e['test'][0])}%", fontsize=10, loc="left", pad=3)
            ax.text(1.0, 1.03, "x".join(str(w) for w in sorted(set(e["widths"]))) + (f" x{d}" if d > 1 else " wide"),
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color=MUTED)
            if i == 0:
                ax.annotate(f"{short(b)} parameters", (0.5, 1.2), xycoords="axes fraction", ha="center", va="bottom",
                            fontsize=9.5, color=INK2)
            if j == 0:
                ax.set_ylabel(f"{d} hidden layer{'s' if d > 1 else ''}", fontsize=9.5, color=INK2)
    save(fig, out, name)


def fig_size(out, nets):
    """Accuracy against the number of parameters, one line per depth."""
    fig, axes = panels()
    for ax, task in zip(axes, MAIN_TASKS):
        t = get_task(task)
        rows = collect(ex.grid((task,)), nets)
        budgets = ex.BUDGETS[t.n_in]
        linear = [r for r in rows if r["depth"] == 0]
        if linear:
            ax.axhline(100 * linear[0]["acc"], color=GREY, linewidth=1.4, zorder=1)
        for depth, color in zip(LINE_DEPTHS, RAMP5):
            pts = sorted((r for r in rows if r["depth"] == depth), key=lambda r: r["params"])
            if not pts:
                continue
            x = [r["params"] for r in pts]
            ax.vlines(x, [100 * r["lo"] for r in pts], [100 * r["hi"] for r in pts], color=color, linewidth=1.6, alpha=0.5)
            ax.plot(x, [100 * r["acc"] for r in pts], markersize=5, label=f"{depth}", **marker(color))
        ax.set_title(title_of(task))
        log_params_axis(ax, budgets[0], budgets[-1])
        acc_axis(ax, t.chance)
    for ax in axes[3:]:
        ax.set_xlabel("parameters (weights and biases)")
    for ax in (axes[0], axes[3]):
        ax.set_ylabel("test accuracy, %")
    handles, labels = axes[1].get_legend_handles_labels()
    handles = [matplotlib.lines.Line2D([], [], color=GREY, linewidth=1.4)] + handles
    labels = ["none"] + labels
    fig.legend(handles, labels, title="hidden layers", loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 1.0),
               title_fontsize=8.5, columnspacing=1.6, handlelength=1.6)
    save(fig, out, "size.png", rect=(0, 0, 1, 0.93))


def _heat(ax, table, row_labels, col_labels, lo, hi, fontsize=7.5):
    shown = np.ma.masked_invalid(table)
    ax.imshow(shown, cmap=SEQ, vmin=lo, vmax=hi, aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(col_labels)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    for (i, j), v in np.ndenumerate(table):
        if np.isfinite(v):
            dark = (v - lo) / max(hi - lo, 1e-9) > 0.55
            ax.text(j, i, f"{v:.1f}" if v < 99.95 else "100", ha="center", va="center", fontsize=fontsize,
                    color="#ffffff" if dark else INK)


def fig_heat(out, nets):
    """Depth against budget, test accuracy in every cell."""
    fig, axes = panels(h=2.6)
    for ax, task in zip(axes, MAIN_TASKS):
        t = get_task(task)
        rows = collect(ex.grid((task,)), nets)
        budgets = ex.BUDGETS[t.n_in]
        table = np.full((len(ex.DEPTHS), len(budgets)), np.nan)
        for r in rows:
            if r["depth"]:
                table[ex.DEPTHS.index(r["depth"]), budgets.index(r["budget"])] = 100 * r["acc"]
        _heat(ax, table, [str(d) for d in ex.DEPTHS], [short(b) for b in budgets], 100 * t.chance, 100)
        ax.set_title(title_of(task))
    for ax in axes[3:]:
        ax.set_xlabel("parameters")
    for ax in (axes[0], axes[3]):
        ax.set_ylabel("hidden layers")
    save(fig, out, "depth_heatmap.png")


def needed(rows, margin=0.01):
    """The smallest network whose accuracy is within `margin` of the best one in rows."""
    if not rows:
        return None, None
    best = max(r["acc"] for r in rows)
    ok = [r for r in rows if r["acc"] >= best - margin]
    return min(ok, key=lambda r: r["params"]), best


def fig_needed(out, nets):
    """How big the smallest good-enough network is, task by task."""
    fig, ax = plt.subplots(figsize=(9.6, 2.9))
    names, values, notes = [], [], []
    for task in MAIN_TASKS:
        rows = [r for r in collect(ex.grid((task,)), nets)]
        small, best = needed(rows)
        if small is None:
            continue
        names.append(title_of(task))
        values.append(small["params"])
        shape = "no hidden layer" if not small["widths"] else f"{len(small['widths'])} x {small['widths'][0]}"
        notes.append(f"{short(small['params'])} parameters ({shape}), {pct(small['acc'])}%")
    y = np.arange(len(names))[::-1]
    ax.hlines(y, 1, values, color=BLUE, linewidth=7, capstyle="round")
    for yy, v, note in zip(y, values, notes):
        ax.text(v * 1.35, yy, note, va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(3, 3e7)
    ax.set_xticks([10, 100, 1000, 10000, 100000])
    ax.set_xticklabels(["10", "100", "1k", "10k", "100k"])
    ax.minorticks_off()
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.set_xlabel("parameters in the smallest network within 1 point of the best accuracy on that task")
    save(fig, out, "needed.png")


def fig_dial(out, nets):
    """One task made harder: spirals with more turns."""
    rows = collect(ex.dial(), nets)
    if not rows:
        return
    turns = [t for t in ex.DIAL_TURNS if any(r["turns"] == t for r in rows)]
    budgets = ex.BUDGETS[2]
    table = np.full((len(turns), len(budgets)), np.nan)
    for r in rows:
        table[turns.index(r["turns"]), budgets.index(r["budget"])] = 100 * r["acc"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 3.3), gridspec_kw=dict(width_ratios=[1.45, 1]))
    _heat(a, table, [f"{t:g}" for t in turns], [short(x) for x in budgets], 50, 100, fontsize=8)
    a.set_title("Test accuracy, %  ·  3 hidden layers")
    a.set_xlabel("parameters")
    a.set_ylabel("turns of the spiral")
    xs, ys = [], []
    for t in turns:
        small, _ = needed([r for r in rows if r["turns"] == t])
        if small:
            xs.append(t)
            ys.append(small["params"])
    b.plot(xs, ys, **marker(BLUE))
    for x, yv in zip(xs, ys):
        b.annotate(short(yv), (x, yv), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, color=INK2)
    b.set_yscale("log")
    b.set_ylim(20, 300000)
    b.set_yticks([100, 1000, 10000, 100000])
    b.set_yticklabels(["100", "1k", "10k", "100k"])
    b.minorticks_off()
    b.set_xticks(turns)
    b.set_xticklabels([f"{t:g}" for t in turns])
    b.set_title("Smallest network within 1 point of the best")
    b.set_xlabel("turns of the spiral")
    b.set_ylabel("parameters")
    save(fig, out, "dial.png")


def fig_sweeps(out, nets):
    """Widen only the first layer, or only the rest."""
    fig, axes = panels()
    for ax, task in zip(axes, MAIN_TASKS):
        t = get_task(task)
        rows = collect(ex.one_layer_sweeps((task,)), nets)
        for exp, field, color, shape, label in (("sweep_first", "first", BLUE, "o", f"first layer swept, the rest {ex.REST}"),
                                                ("sweep_rest", "rest", ORANGE, "s", f"later layers swept, the first {ex.REST}")):
            pts = sorted((r for r in rows if r["exp"] == exp), key=lambda r: r[field])
            if not pts:
                continue
            x = [r[field] for r in pts]
            ax.vlines(x, [100 * r["lo"] for r in pts], [100 * r["hi"] for r in pts], color=color, linewidth=1.6, alpha=0.5)
            ax.plot(x, [100 * r["acc"] for r in pts], markersize=5, label=label, **marker(color, shape))
            r = pts[-1]                                   # how many parameters the biggest network on the line has
            ax.annotate(short(r["params"]), (r[field], 100 * r["acc"]), textcoords="offset points",
                        xytext=(0, 8 if exp == "sweep_first" else -14), ha="center", fontsize=7.5, color=INK2)
        ax.set_title(title_of(task))
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 4, 16, 64, 256])
        ax.set_xticklabels(["1", "4", "16", "64", "256"])
        ax.set_xlim(0.75, 700)
        ax.minorticks_off()
        acc_axis(ax, t.chance)
    for ax in axes[3:]:
        ax.set_xlabel("width of the swept layer(s)")
    for ax in (axes[0], axes[3]):
        ax.set_ylabel("test accuracy, %")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0), columnspacing=2.5)
    save(fig, out, "first_layer_sweeps.png", rect=(0, 0, 1, 0.95))


def _two_budget_legend(fig):
    handles = [matplotlib.lines.Line2D([], [], linewidth=2, markersize=6, **marker(LIGHT_BLUE, "o")),
               matplotlib.lines.Line2D([], [], linewidth=2, markersize=6, **marker(DARK_BLUE, "s"))]
    fig.legend(handles, ["the smaller budget", "ten times as many parameters"], loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.0), columnspacing=2.5)


def _budget_title(task):
    lo, hi = ex.TIGHT_ROOMY[task]
    return f"{title_of(task)}  ·  {short(lo)} and {short(hi)}"


def fig_first_budget(out, nets):
    """A fixed budget: how much of it should the first layer get?"""
    fig, axes = panels()
    for ax, task in zip(axes, MAIN_TASKS):
        t = get_task(task)
        rows = collect(ex.first_budget((task,)), nets)
        for level, color, shape in (("tight", LIGHT_BLUE, "o"), ("roomy", DARK_BLUE, "s")):
            pts = sorted((r for r in rows if r["level"] == level), key=lambda r: r["first"])
            if not pts:
                continue
            x = [100 * first_layer_share(t.n_in, r["widths"], t.n_out) for r in pts]
            ax.vlines(x, [100 * r["lo"] for r in pts], [100 * r["hi"] for r in pts], color=color, linewidth=1.6, alpha=0.5)
            ax.plot(x, [100 * r["acc"] for r in pts], markersize=5, **marker(color, shape))
        ax.set_title(_budget_title(task))
        ax.set_xlim(-4, 104)
        ax.set_xticks([0, 25, 50, 75, 100])
        acc_axis(ax, t.chance)
    for ax in axes[3:]:
        ax.set_xlabel("weights feeding the first hidden layer, % of all")
    for ax in (axes[0], axes[3]):
        ax.set_ylabel("test accuracy, %")
    _two_budget_legend(fig)
    save(fig, out, "first_layer_budget.png", rect=(0, 0, 1, 0.95))


def fig_shapes(out, nets):
    """Five shapes, same number of parameters."""
    fig, axes = panels(h=2.5)
    names = list(ex.SHAPES)
    for ax, task in zip(axes, MAIN_TASKS):
        rows = collect(ex.shapes((task,)), nets)
        values = []
        for level, color, shape, dy in (("tight", LIGHT_BLUE, "o", 0.16), ("roomy", DARK_BLUE, "s", -0.16)):
            for r in rows:
                if r["level"] != level:
                    continue
                y = len(names) - 1 - names.index(r["shape"]) + dy
                ax.hlines(y, 100 * r["lo"], 100 * r["hi"], color=color, linewidth=2, alpha=0.45)
                ax.plot([100 * r["acc"]], [y], linestyle="none", markersize=6.5, **marker(color, shape))
                values += [100 * r["lo"], 100 * r["hi"]]
        ax.set_title(_budget_title(task))
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names[::-1] if ax in (axes[0], axes[3]) else [""] * len(names), color=INK, fontsize=8.5)
        ax.set_ylim(-0.6, len(names) - 0.4)
        ax.grid(axis="y", visible=False)
        if values:
            lo, hi = min(values), max(values)
            pad = max(1.0, (hi - lo) * 0.12)
            ax.set_xlim(max(0, lo - pad), min(100.5, hi + pad))
    for ax in axes[3:]:
        ax.set_xlabel("test accuracy, %")
    _two_budget_legend(fig)
    save(fig, out, "shapes.png", rect=(0, 0, 1, 0.95))


def fig_activations(out, nets, depth, name):
    fig, axes = panels()
    for ax, task in zip(axes, MAIN_TASKS):
        t = get_task(task)
        rows = [r for r in collect(ex.activations((task,)), nets) if r["depth"] == depth]
        budgets = [b for b in ex.BUDGETS[t.n_in] if b <= ex.ACTIVATION_BUDGET_CAP]
        for act in ex.ACTIVATIONS:
            pts = sorted((r for r in rows if r["activation"] == act), key=lambda r: r["params"])
            if pts:
                ax.plot([r["params"] for r in pts], [100 * r["acc"] for r in pts], markersize=5, label=ACT_NAME[act],
                        **marker(ACT_COLOR[act], ACT_MARK[act]))
        ax.set_title(title_of(task))
        log_params_axis(ax, budgets[0], budgets[-1])
        acc_axis(ax, t.chance)
    for ax in axes[3:]:
        ax.set_xlabel("parameters")
    for ax in (axes[0], axes[3]):
        ax.set_ylabel("test accuracy, %")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0), columnspacing=2.2,
               title=f"{depth} hidden layers", title_fontsize=8.5)
    save(fig, out, name, rect=(0, 0, 1, 0.93))


def fig_gap(out, nets, depth=3):
    """Training accuracy against test accuracy as the network grows."""
    fig, axes = panels()
    for ax, task in zip(axes, MAIN_TASKS):
        t = get_task(task)
        pts = sorted((r for r in collect(ex.grid((task,)), nets) if r["depth"] == depth), key=lambda r: r["params"])
        budgets = ex.BUDGETS[t.n_in]
        if pts:
            x = [r["params"] for r in pts]
            ax.plot(x, [100 * r["train"] for r in pts], markersize=5, label="examples it was trained on", **marker(GREY, "s"))
            ax.plot(x, [100 * r["acc"] for r in pts], markersize=5, label="examples it has never seen", **marker(BLUE))
        ax.set_title(title_of(task))
        log_params_axis(ax, budgets[0], budgets[-1])
        acc_axis(ax, t.chance)
    for ax in axes[3:]:
        ax.set_xlabel("parameters")
    for ax in (axes[0], axes[3]):
        ax.set_ylabel("accuracy, %")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0), columnspacing=2.5)
    save(fig, out, "train_vs_test.png", rect=(0, 0, 1, 0.95))


def fig_filters(out, nets, firsts=(2, 8, 64)):
    """What the first layer's units look at on the big digit task."""
    rows = {r["first"]: r for r in collect(ex.one_layer_sweeps(("digits1000",)), nets) if r["exp"] == "sweep_first" and "filters" in r}
    firsts = [f for f in firsts if f in rows]
    if not firsts:
        return
    show = 8
    fig, axes = plt.subplots(len(firsts), show, figsize=(9.6, 1.9 * len(firsts)), squeeze=False)
    for i, f in enumerate(firsts):
        r = rows[f]
        for j in range(show):
            ax = axes[i][j]
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if j < len(r["filters"]):
                w = r["filters"][j].reshape(40, 25)
                top = np.abs(w).max() or 1.0
                ax.imshow(w, cmap=DIV, vmin=-top, vmax=top, interpolation="nearest")
            if j == 0:
                ax.set_ylabel(f"first layer {f} wide\n{pct(r['acc'])}%", fontsize=8.5, color=INK2)
    save(fig, out, "first_layer_filters.png")


def fig_cost(out, cost):
    """What a network costs to run and train, and where its weights sit."""
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(9.6, 3.2))
    series = [(2, BLUE, "o", "2 inputs"), (96, ORANGE, "s", "96 inputs"), (1000, AQUA, "^", "1,000 inputs")]
    for ax, field, title, unit in ((a, "train_ms", "One training step (128 examples)", "milliseconds"),
                                   (b, "infer_us", "Answering one example", "microseconds")):
        for n_in, color, shape, label in series:
            pts = sorted((r for r in cost if r["n_in"] == n_in and r["depth"] == 3 and r["kind"] == "flat"), key=lambda r: r["params"])
            if pts:
                ax.plot([r["params"] for r in pts], [r[field] for r in pts], markersize=5, label=label, **marker(color, shape))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([100, 1000, 10000, 100000])
        ax.set_xticklabels(["100", "1k", "10k", "100k"])
        ticks = [1, 2, 5, 10] if field == "train_ms" else [0.3, 1, 3, 10]
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{v:g}" for v in ticks])
        ax.minorticks_off()
        ax.set_title(title)
        ax.set_xlabel("parameters  ·  3 hidden layers")
        ax.set_ylabel(unit)
    a.legend(loc="upper left")
    widths = np.array([4, 8, 16, 32, 64, 128, 256, 512, 1024])
    for n_in, color, shape, label in series:
        n_out = 10 if n_in != 2 else 2
        c.plot(widths, [100 * first_layer_share(n_in, (w, w, w), n_out) for w in widths], markersize=5, label=label, **marker(color, shape))
    c.set_xscale("log", base=2)
    c.set_xticks([4, 16, 64, 256, 1024])
    c.set_xticklabels(["4", "16", "64", "256", "1,024"])
    c.minorticks_off()
    c.set_ylim(-4, 104)
    c.set_title("Weights that feed the first layer")
    c.set_xlabel("width of each of 3 hidden layers")
    c.set_ylabel("% of all weights")
    save(fig, out, "cost.png")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def cell(r):
    """The middle seed's accuracy, with the lowest and highest in brackets when they are a point or more apart."""
    if r["hi"] - r["lo"] >= 0.01:
        return f"{pct(r['acc'])} ({np.floor(100 * r['lo']):.0f}-{np.ceil(100 * r['hi']):.0f})"
    return pct(r["acc"])


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def write_results(out, nets, cost, path):
    L = []
    add = L.append
    add("# Results in full\n")
    add("Generated by `report.py` from the files in `results/runs`. Every accuracy is the test accuracy in %: "
        "the middle one of three seeds (the median) at the network's best learning rate. Where the lowest and highest seed "
        "are a point or more apart they follow in brackets.\n")
    total_runs = sum(n["runs"] for n in nets.values())
    add(f"{len(nets)} networks, {total_runs:,} training runs.\n")

    add("## Size and depth\n")
    add("Equal-width networks. Each cell: accuracy, then the width of every hidden layer in brackets.\n")
    for task in MAIN_TASKS:
        t = get_task(task)
        rows = collect(ex.grid((task,)), nets)
        budgets = ex.BUDGETS[t.n_in]
        add(f"### {title_of(task)}\n")
        linear = [r for r in rows if r["depth"] == 0]
        if linear:
            add(f"No hidden layer at all ({linear[0]['params']:,} parameters): {cell(linear[0])}. Chance is {pct(t.chance, 0)}.\n")
        body = []
        for d in ex.DEPTHS:
            line = [f"{d}"]
            for b in budgets:
                r = [x for x in rows if x["depth"] == d and x["budget"] == b]
                line.append(f"{cell(r[0])} ({r[0]['widths'][0]})" if r else "")
            body.append(line)
        add(table(["hidden layers"] + [short(b) for b in budgets], body))
        small, best = needed(rows)
        if small:
            add(f"Best: {pct(best)}. Smallest network within 1 point of it: {small['params']:,} parameters, "
                f"hidden widths {list(small['widths'])} ({pct(small['acc'])}).\n")

    add("## Train against test (3 hidden layers)\n")
    for task in MAIN_TASKS:
        pts = sorted((r for r in collect(ex.grid((task,)), nets) if r["depth"] == 3), key=lambda r: r["params"])
        add(f"**{title_of(task)}**\n")
        add(table(["parameters", "train", "test", "gap"], [[f"{r['params']:,}", pct(r["train"]), pct(r["acc"]), f"{100 * (r['train'] - r['acc']):.1f}"] for r in pts]))

    add("## One task made harder: spirals with more turns (3 hidden layers)\n")
    rows = collect(ex.dial(), nets)
    if rows:
        budgets = ex.BUDGETS[2]
        body = []
        for turns in ex.DIAL_TURNS:
            line = [f"{turns:g}"]
            for b in budgets:
                r = [x for x in rows if x["turns"] == turns and x["budget"] == b]
                line.append(cell(r[0]) if r else "")
            small, _ = needed([x for x in rows if x["turns"] == turns])
            line.append(f"{small['params']:,}" if small else "")
            body.append(line)
        add(table(["turns"] + [short(b) for b in budgets] + ["smallest within 1 point of the best"], body))

    add("## The first layer\n")
    add(f"### Only the first layer changes: [h, {ex.REST}, {ex.REST}]\n")
    for task in MAIN_TASKS:
        t = get_task(task)
        pts = sorted((r for r in collect(ex.one_layer_sweeps((task,)), nets) if r["exp"] == "sweep_first"), key=lambda r: r["first"])
        add(f"**{title_of(task)}**\n")
        add(table(["first layer", "parameters", "in the first layer", "test"],
                  [[r["first"], f"{r['params']:,}", f"{100 * first_layer_share(t.n_in, r['widths'], t.n_out):.0f}%", cell(r)] for r in pts]))
    add(f"### Only the later layers change: [{ex.REST}, w, w]\n")
    for task in MAIN_TASKS:
        t = get_task(task)
        pts = sorted((r for r in collect(ex.one_layer_sweeps((task,)), nets) if r["exp"] == "sweep_rest"), key=lambda r: r["rest"])
        add(f"**{title_of(task)}**\n")
        add(table(["later layers", "parameters", "in the first layer", "test"],
                  [[r["rest"], f"{r['params']:,}", f"{100 * first_layer_share(t.n_in, r['widths'], t.n_out):.0f}%", cell(r)] for r in pts]))
    add("### A fixed budget, split differently: [first, w, w]\n")
    for task in MAIN_TASKS:
        t = get_task(task)
        rows = collect(ex.first_budget((task,)), nets)
        for level in ("tight", "roomy"):
            pts = sorted((r for r in rows if r["level"] == level), key=lambda r: r["first"])
            if not pts:
                continue
            add(f"**{title_of(task)}, about {short(pts[0]['budget'])} parameters**\n")
            add(table(["hidden widths", "parameters", "in the first layer", "test"],
                      [["-".join(map(str, r["widths"])), f"{r['params']:,}", f"{100 * first_layer_share(t.n_in, r['widths'], t.n_out):.0f}%", cell(r)] for r in pts]))

    add("## Five shapes with the same number of parameters\n")
    for task in MAIN_TASKS:
        t = get_task(task)
        rows = collect(ex.shapes((task,)), nets)
        body = []
        for name in ex.SHAPES:
            line = [name]
            for level in ("tight", "roomy"):
                r = [x for x in rows if x["shape"] == name and x["level"] == level]
                line += ["-".join(map(str, r[0]["widths"])), cell(r[0])] if r else ["", ""]
            body.append(line)
        lo, hi = ex.TIGHT_ROOMY[task]
        add(f"**{title_of(task)}**\n")
        add(table(["shape", f"widths at {short(lo)}", "test", f"widths at {short(hi)}", "test"], body))

    add("## Activation functions\n")
    for depth in ex.ACTIVATION_DEPTHS:
        add(f"### {depth} hidden layers\n")
        for task in MAIN_TASKS:
            t = get_task(task)
            rows = [r for r in collect(ex.activations((task,)), nets) if r["depth"] == depth]
            budgets = [b for b in ex.BUDGETS[t.n_in] if b <= ex.ACTIVATION_BUDGET_CAP]
            body = []
            for act in ex.ACTIVATIONS:
                line = [ACT_NAME[act]]
                for b in budgets:
                    r = [x for x in rows if x["activation"] == act and x["budget"] == b]
                    line.append(cell(r[0]) if r else "")
                body.append(line)
            add(f"**{title_of(task)}**\n")
            add(table(["activation"] + [short(b) for b in budgets], body))

    add("## ReLU units that never fire\n")
    add("Share of hidden units that output zero for every test example, ReLU networks, mean of three seeds.\n")
    for task in MAIN_TASKS:
        t = get_task(task)
        rows = collect(ex.grid((task,)), nets)
        budgets = ex.BUDGETS[t.n_in]
        body = []
        for d in ex.DEPTHS:
            line = [f"{d}"]
            for b in budgets:
                r = [x for x in rows if x["depth"] == d and x["budget"] == b]
                line.append(f"{100 * r[0]['dead']:.0f}%" if r else "")
            body.append(line)
        add(f"**{title_of(task)}**\n")
        add(table(["hidden layers"] + [short(b) for b in budgets], body))

    add("## Runs that collapsed\n")
    add("A run is counted as collapsed when it ended at chance level (within 3 points) although the same network scored "
        "at least 15 points above chance in another run. \"Kept runs\" are the three seeds at the learning rate that won; "
        "\"all runs\" adds the rest of the learning-rate search, which includes rates that were too high.\n")
    counts = {}
    for n in nets.values():
        group = counts.setdefault((n["activation"], len(n["widths"])), [0, 0, 0, 0, 0])
        group[0] += 1
        group[1] += len(n["test"])
        group[2] += sum(1 for c in n["collapsed"] if c[2])
        group[3] += n["runs"]
        group[4] += len(n["collapsed"])
    order = list(ACT_NAME)
    body = [[ACT_NAME[act], depth] + c for (act, depth), c in sorted(counts.items(), key=lambda kv: (order.index(kv[0][0]), kv[0][1]))]
    add(table(["activation", "hidden layers", "networks", "kept runs", "collapsed", "all runs", "collapsed"], body))
    kept = sum(c[2] for c in counts.values())
    relu = sum(c[2] for (act, _), c in counts.items() if act == "relu")
    pinched = sum(1 for n in nets.values() for c in n["collapsed"] if c[2] and n["widths"] and min(n["widths"]) <= 3)
    add(f"{kept} of the {sum(c[1] for c in counts.values()):,} kept runs collapsed, {relu} of them in ReLU networks; "
        f"{pinched} of the {kept} were in networks with a hidden layer of 3 units or fewer. "
        f"Over all {total_runs:,} runs, {sum(c[4] for c in counts.values())} collapsed.\n")

    add("## Learning rates\n")
    add("Which learning rate each network ended up with. The search starts with three rates and carries on past either end "
        "while the best rate found is the last one tried.\n")
    ladder = list(ex.LR_LADDER)
    counts = {}
    for n in nets.values():
        group = "under 30,000 parameters (starts with 0.003, 0.01, 0.03)" if n["params"] < 30000 else "30,000 parameters or more (starts with 0.001, 0.003, 0.01)"
        counts.setdefault(group, [0] * len(ladder))[ladder.index(n["lr"])] += 1
    add(table(["networks"] + [f"{lr:g}" for lr in ladder], [[g] + c for g, c in sorted(counts.items(), reverse=True)]))
    extended = sum(1 for n in nets.values() if len(n["search"]) > 3)
    add(f"{extended} of {len(nets)} networks had their search extended beyond the first three rates.\n")
    gaps = [max(x[2] for x in n["search"]) - min(x[2] for x in n["search"]) for n in nets.values()]
    add(f"Across the learning rates tried on seed 0, the gap in test accuracy between the best and the worst was "
        f"{100 * float(np.median(gaps)):.1f} points for the median network and more than 5 points for "
        f"{100 * float(np.mean([g > 0.05 for g in gaps])):.0f}% of them.\n")

    if cost:
        add("## Cost\n")
        add("Measured by `bench.py` on one core with ordinary PyTorch (autograd and `torch.optim.Adam`), one network at a time. "
            "A training step is one batch of 128 examples. Inference is per example, in batches of 1,000.\n")
        for n_in in (2, 96, 1000):
            pts = sorted((r for r in cost if r["n_in"] == n_in and r["kind"] == "flat"), key=lambda r: (r["depth"], r["params"]))
            if not pts:
                continue
            add(f"**{n_in:,} inputs**\n")
            add(table(["hidden layers", "width", "parameters", "training step, ms", "one example, microseconds"],
                      [[r["depth"], r["widths"][0] if r["widths"] else "-", f"{r['params']:,}", f"{r['train_ms']:.2f}", f"{r['infer_us']:.2f}"] for r in pts]))
        wide = sorted((r for r in cost if r["kind"] == "wide"), key=lambda r: r["params"])
        if wide:
            add("**One very wide hidden layer on 2 inputs, against 3 layers with the same number of parameters**\n")
            add(table(["hidden widths", "parameters", "training step, ms", "one example, microseconds"],
                      [["-".join(map(str, r["widths"])), f"{r['params']:,}", f"{r['train_ms']:.2f}", f"{r['infer_us']:.2f}"] for r in wide]))

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("  wrote", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    style()
    nets = load(a.out)
    cost = load_cost(a.out)
    print(f"{len(nets)} networks loaded")
    fig_tasks(a.out)
    fig_maps(a.out, nets)
    fig_size(a.out, nets)
    fig_heat(a.out, nets)
    fig_needed(a.out, nets)
    fig_dial(a.out, nets)
    fig_sweeps(a.out, nets)
    fig_first_budget(a.out, nets)
    fig_shapes(a.out, nets)
    fig_activations(a.out, nets, 3, "activations_3_layers.png")
    fig_activations(a.out, nets, 8, "activations_8_layers.png")
    fig_gap(a.out, nets)
    fig_filters(a.out, nets)
    if cost:
        fig_cost(a.out, cost)
    write_results(a.out, nets, cost, "RESULTS.md" if a.out == "results" else os.path.join(a.out, "RESULTS.md"))


if __name__ == "__main__":
    main()
