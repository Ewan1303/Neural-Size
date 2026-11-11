"""What gets trained: every network in every experiment.

Each experiment is a list of entries, and an entry names a task, a list of
hidden-layer widths and an activation function, plus a few labels saying why it
is there (its depth, its parameter budget, the name of its shape ...). The same
network can appear in several experiments; it is only trained once.

How one network is trained (see train_network below): three learning rates on
seed 0; if the best of them (by validation accuracy) is the highest or the
lowest, the next rate in that direction is tried too, and so on until the best
is no longer at an end; then two more seeds at the rate that won. So every
network costs five or more training runs and reports three.
"""
import base64

import numpy as np

from .nets import count_params, first_layer_shape, flat_shape, scaled_shape
from .tasks import MAIN_TASKS, get_task

SEEDS = (0, 1, 2)
DEPTHS = (1, 2, 3, 5, 8, 12)
MAX_WIDTH = 4096                 # shapes with a wider layer than this are left out (see README, "Cost")

BUDGETS = {
    2: (30, 100, 300, 1000, 3000, 10000, 30000, 100000),
    96: (300, 1000, 3000, 10000, 30000, 100000),
    1000: (3000, 10000, 30000, 100000, 300000),
}

# Two budgets per task for the shape experiments: one where the task is still
# short of weights and one ten times bigger.
TIGHT_ROOMY = {
    "circle": (100, 1000), "spirals": (300, 3000), "digits96": (3000, 30000),
    "needle": (10000, 100000), "sensors": (10000, 100000), "digits1000": (10000, 100000),
}

SHAPES = {                       # three hidden layers in these proportions
    "flat": (1, 1, 1),
    "funnel": (4, 2, 1),
    "reverse funnel": (1, 2, 4),
    "hourglass": (4, 1, 4),
    "diamond": (1, 4, 1),
}

SWEEP_WIDTHS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
SWEEP_REST_WIDTHS = SWEEP_WIDTHS[:-1]
REST = 64                        # the width of the layers that are held fixed in the one-layer-at-a-time sweeps
ACTIVATIONS = ("relu", "leaky_relu", "gelu", "tanh")
ACTIVATION_DEPTHS = (3, 8)
ACTIVATION_BUDGET_CAP = 100000   # the activation comparison stops here to keep the whole run to about an hour
DIAL_TURNS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0)
DIAL_DEPTH = 3


LR_LADDER = (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1)


def lr_grid(params):
    """The three peak learning rates a network of this size is tried at first."""
    return (0.003, 0.01, 0.03) if params < 30000 else (0.001, 0.003, 0.01)


def next_lr(tried, best):
    """The next learning rate to try, or None when the best one tried is not at an end of the range
    (or the ladder has run out)."""
    lo, hi = min(tried), max(tried)
    if best == hi and LR_LADDER.index(hi) + 1 < len(LR_LADDER):
        return LR_LADDER[LR_LADDER.index(hi) + 1]
    if best == lo and LR_LADDER.index(lo) > 0:
        return LR_LADDER[LR_LADDER.index(lo) - 1]
    return None


def key(task, widths, activation="relu"):
    return f"{task}|{activation}|{'-'.join(str(w) for w in widths) or 'linear'}"


def _entry(exp, task, widths, activation="relu", **labels):
    t = get_task(task)
    widths = tuple(int(w) for w in widths)
    return dict(exp=exp, task=task, widths=widths, activation=activation, key=key(task, widths, activation),
                params=count_params(t.n_in, widths, t.n_out), **labels)


def _ok(widths):
    return widths is not None and max(widths) <= MAX_WIDTH


# ----------------------------------------------------------------------------
# The experiments
# ----------------------------------------------------------------------------

def grid(tasks=MAIN_TASKS):
    """Size and depth: equal-width networks of every depth at every parameter budget, plus no hidden layer at all."""
    out = []
    for task in tasks:
        t = get_task(task)
        out.append(_entry("grid", task, (), depth=0, budget=None))
        for depth in DEPTHS:
            for budget in BUDGETS[t.n_in]:
                if budget >= 300000 and depth in (5, 12):
                    continue                              # the costliest column is run at four depths only
                widths = flat_shape(t.n_in, t.n_out, depth, budget)
                if _ok(widths):
                    out.append(_entry("grid", task, widths, depth=depth, budget=budget))
    return out


def first_budget(tasks=MAIN_TASKS):
    """A fixed number of parameters, split differently: [first, w, w] with the first layer swept
    from a single unit to nearly the whole budget, and w taking whatever is left."""
    out = []
    for task in tasks:
        t = get_task(task)
        for level, budget in zip(("tight", "roomy"), TIGHT_ROOMY[task]):
            shapes_ = {}
            h = 1
            while True:                                   # first layer 1, 2, 4, 8 ... while the budget holds
                widths = first_layer_shape(t.n_in, t.n_out, h, budget, tolerance=0.12)
                if widths is None:
                    if count_params(t.n_in, (h, 2, 2), t.n_out) > budget * 1.12:
                        break
                else:
                    shapes_[h] = widths
                h *= 2
            top = max(shapes_)                            # ... and the widest first layer that still leaves
            while True:                                   # the later layers 8 units each
                widths = first_layer_shape(t.n_in, t.n_out, top + 1, budget, tolerance=0.05, min_width=8)
                if widths is None or count_params(t.n_in, widths, t.n_out) > budget * 1.05:
                    break
                top += 1
            if top > max(shapes_):
                shapes_[top] = first_layer_shape(t.n_in, t.n_out, top, budget, tolerance=0.05, min_width=8)
            for h, widths in sorted(shapes_.items()):
                if _ok(widths):
                    out.append(_entry("first_budget", task, widths, budget=budget, level=level, first=h))
    return out


def one_layer_sweeps(tasks=MAIN_TASKS):
    """Change one thing at a time: [h, 64, 64] with only the first layer swept, and [64, w, w] with only the rest swept."""
    out = []
    for task in tasks:
        for h in SWEEP_WIDTHS:
            out.append(_entry("sweep_first", task, (h, REST, REST), first=h))
        for h in SWEEP_REST_WIDTHS:
            out.append(_entry("sweep_rest", task, (REST, h, h), rest=h))
    return out


def shapes(tasks=MAIN_TASKS):
    """Five named three-layer shapes with the same number of parameters."""
    out = []
    for task in tasks:
        t = get_task(task)
        for level, budget in zip(("tight", "roomy"), TIGHT_ROOMY[task]):
            for name, ratios in SHAPES.items():
                widths = scaled_shape(t.n_in, t.n_out, ratios, budget)
                if _ok(widths):
                    out.append(_entry("shapes", task, widths, budget=budget, level=level, shape=name))
    return out


def activations(tasks=MAIN_TASKS):
    """The size sweep again with other activation functions, on a shallow (3) and a deep (8) network."""
    out = []
    for task in tasks:
        t = get_task(task)
        for depth in ACTIVATION_DEPTHS:
            for budget in BUDGETS[t.n_in]:
                widths = flat_shape(t.n_in, t.n_out, depth, budget)
                if not _ok(widths) or budget > ACTIVATION_BUDGET_CAP:
                    continue
                for act in ACTIVATIONS:
                    out.append(_entry("activations", task, widths, act, depth=depth, budget=budget))
    return out


def dial():
    """One task made steadily harder: spirals with more and more turns, at every budget."""
    out = []
    for turns in DIAL_TURNS:
        task = "spirals" if turns == 3.0 else f"spirals@{turns:g}"
        t = get_task(task)
        for budget in BUDGETS[2]:
            widths = flat_shape(t.n_in, t.n_out, DIAL_DEPTH, budget)
            if _ok(widths):
                out.append(_entry("dial", task, widths, turns=turns, budget=budget, depth=DIAL_DEPTH))
    return out


EXPERIMENTS = {"grid": grid, "first_budget": first_budget, "sweeps": one_layer_sweeps, "shapes": shapes,
               "activations": activations, "dial": dial}


def all_entries():
    out = []
    for make in EXPERIMENTS.values():
        out.extend(make())
    return out


def unique_networks(entries):
    """One job per distinct (task, widths, activation), remembering which experiments want it."""
    jobs = {}
    for e in entries:
        job = jobs.setdefault(e["key"], dict(key=e["key"], task=e["task"], widths=e["widths"],
                                             activation=e["activation"], params=e["params"], exps=[]))
        if e["exp"] not in job["exps"]:
            job["exps"].append(e["exp"])
    return list(jobs.values())


def estimate_seconds(job, steps_scale=1.0, seeds=SEEDS):
    """A rough guess at the CPU time one network's training runs will take (used for the progress estimate)."""
    t = get_task(job["task"])
    widths = job["widths"]
    runs = len(lr_grid(job["params"])) + len(seeds)          # the three first rates, about one more, the other seeds
    per_step = 2.5e-8 * job["params"] + 3e-7 * sum(widths)
    sizes = [t.n_in, *widths, t.n_out]
    for a, b, c in zip(sizes[:-2], sizes[1:-1], sizes[2:]):       # a huge layer between two tiny ones is slow per weight
        if b >= 256 and min(a, c) <= 16:
            per_step += 1.3e-6 * b
    overhead = 2 * (2e-4 + 1e-4 * (len(widths) + 1))
    return t.steps * steps_scale * (runs * per_step + overhead) + 0.5


# ----------------------------------------------------------------------------
# Training one network (five runs)
# ----------------------------------------------------------------------------

def _better(a, b):
    """True when run a beats run b on the validation set (accuracy first, then loss)."""
    la = a["val_loss"] if a["val_loss"] is not None else float("inf")
    lb = b["val_loss"] if b["val_loss"] is not None else float("inf")
    return (a["val_acc"], -la) > (b["val_acc"], -lb)


def _pack(r, want_filters):
    """Turn a run's pictures (numpy arrays) into text so the record can be saved as JSON."""
    r = dict(r)
    picture = r.pop("map", None)
    filters = r.pop("filters", None)
    if isinstance(picture, str):
        r["map"] = picture
    elif picture is not None:
        r["map"] = base64.b64encode(picture.tobytes()).decode()
    if isinstance(filters, str):
        r["filters"] = filters
    elif filters is not None and want_filters:
        f = np.asarray(filters[:16], dtype=np.float32)
        scale = float(np.abs(f).max()) or 1.0
        r["filters"] = base64.b64encode(np.round(f / scale * 127).astype(np.int8).tobytes()).decode()
        r["filters_shape"] = list(f.shape)
    return r


def best_run(search):
    best = search[0]
    for r in search[1:]:
        if _better(r, best):
            best = r
    return best


def search_is_open(records):
    """True when a saved network's learning-rate search stopped with its best rate at an end of the range."""
    search = [r for r in records if r["role"] == "search"]
    return next_lr([r["lr"] for r in search], best_run(search)["lr"]) is not None


def train_network(job, steps_scale=1.0, seeds=SEEDS, previous=None):
    """Train one network the standard way and return its run records.

    Learning-rate search on the first seed (three rates, extended while the best
    is at an end of the range), then the other seeds at the winning rate. Pass a
    network's earlier records as `previous` to carry on from them rather than start again.
    """
    from .fast import train_group              # imported here so that listing the experiments needs no PyTorch

    t = get_task(job["task"])
    steps = max(24, int(round(t.steps * steps_scale)))
    want_filters = job["task"] == "digits1000" and "sweep_first" in job["exps"]
    extras = t.grid or want_filters
    args = (job["task"], job["widths"], job["activation"])

    previous = previous or []
    search = [dict(r) for r in previous if r["role"] == "search"]
    if not search:
        lrs = list(lr_grid(job["params"]))
        search = [_pack(r, want_filters) for r in train_group(*args, lrs, [seeds[0]] * len(lrs), steps=steps, extras=extras)]
    while True:
        best = best_run(search)
        more = next_lr([r["lr"] for r in search], best["lr"])
        if more is None:
            break
        search += [_pack(r, want_filters) for r in train_group(*args, [more], [seeds[0]], steps=steps, extras=extras)]
    rest = [dict(r) for r in previous if r["role"] == "seed" and r["lr"] == best["lr"]]
    if len(rest) != len(seeds) - 1:
        rest = train_group(*args, [best["lr"]] * (len(seeds) - 1), list(seeds[1:]), steps=steps) if len(seeds) > 1 else []
    records = []
    for i, r in enumerate(search + rest):
        r = dict(r)
        r["key"] = job["key"]
        r["role"] = "search" if i < len(search) else "seed"
        r["chosen"] = r["lr"] == best["lr"]
        if not (r["chosen"] and r["role"] == "search"):        # pictures are kept for one run only
            for k in ("map", "filters", "filters_shape"):
                r.pop(k, None)
        records.append(r)
    return records
