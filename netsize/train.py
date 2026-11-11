"""Train one network on one task and measure what came out.

Every network gets the same treatment: the same training examples, the same
number of steps, batches of 128, the Adam optimiser, and a learning rate that
warms up for the first 5% of the steps and then falls smoothly to zero. The
only things that change from run to run are the network and the peak learning
rate.
"""
import math

import torch
import torch.nn.functional as F

from .clock import cpu_seconds
from .nets import MLP, count_params, weights_per_layer
from .tasks import get_task

BATCH = 128
WARMUP = 0.05
CHECKPOINTS = 24
GRID = 64            # decision maps are GRID x GRID pictures
GRID_SPAN = 2.2      # ... covering -2.2..2.2 on both inputs


def _schedule(step, steps, peak):
    warm = max(1, int(round(WARMUP * steps)))
    if step < warm:
        return peak * (step + 1) / warm
    return 0.5 * peak * (1.0 + math.cos(math.pi * (step - warm) / max(1, steps - warm)))


@torch.no_grad()
def evaluate(net, x, y, chunk=4096):
    """Accuracy and mean cross-entropy loss of the network on a set of examples."""
    right, loss = 0, 0.0
    for i in range(0, len(x), chunk):
        out = net(x[i:i + chunk])
        right += int((out.argmax(1) == y[i:i + chunk]).sum())
        loss += float(F.cross_entropy(out, y[i:i + chunk], reduction="sum"))
    return right / len(x), loss / len(x)


@torch.no_grad()
def dead_units(net, x, chunk=4096):
    """Fraction of hidden units that give exactly zero for every example in x (a ReLU that never fires)."""
    alive = None
    for i in range(0, len(x), chunk):
        hidden = []
        net(x[i:i + chunk], hidden)
        fired = [(h != 0).any(0) for h in hidden]
        alive = fired if alive is None else [a | f for a, f in zip(alive, fired)]
    total = sum(a.numel() for a in alive)
    if total == 0:                                        # no hidden layer at all
        return 0.0
    return 1.0 - sum(int(a.sum()) for a in alive) / total


@torch.no_grad()
def decision_map(net):
    """The network's answer over a square of the two inputs: probability of class 1, as bytes 0..255."""
    axis = torch.linspace(-GRID_SPAN, GRID_SPAN, GRID)
    gy, gx = torch.meshgrid(axis, axis, indexing="ij")
    p = torch.softmax(net(torch.stack([gx.reshape(-1), gy.reshape(-1)], 1)), 1)[:, 1]
    return (p * 255).round().to(torch.uint8).reshape(GRID, GRID)


def train(task_name, widths, activation="relu", lr=1e-3, seed=0, steps=None, extras=False):
    """Train a network with the given hidden widths; return a dictionary of measurements.

    extras=True also returns things for pictures: the decision map on two-input
    tasks and the first hidden layer's incoming weights on the big digit task.
    """
    task = get_task(task_name)
    data = task.data(seed)
    steps = int(steps or task.steps)
    net = MLP(task.n_in, widths, task.n_out, activation, seed=seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    n = len(data.train_x)
    shuffle = torch.Generator().manual_seed(1000 * seed + 17)
    order = torch.randperm(n, generator=shuffle)
    at = 0
    marks = sorted({int(round(steps * (k + 1) / CHECKPOINTS)) for k in range(CHECKPOINTS)})
    curve = []                                     # (step, CPU seconds so far, validation accuracy)
    spent = 0.0
    t0 = cpu_seconds()
    diverged = False
    for step in range(steps):
        if at + BATCH > n:
            order, at = torch.randperm(n, generator=shuffle), 0
        idx = order[at:at + BATCH]
        at += BATCH
        for group in opt.param_groups:
            group["lr"] = _schedule(step, steps, lr)
        loss = F.cross_entropy(net(data.train_x[idx]), data.train_y[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step + 1 == marks[len(curve)]:
            spent += cpu_seconds() - t0             # the clock stops while the network is being tested
            if not math.isfinite(loss.item()):
                diverged = True
                break
            curve.append((step + 1, round(spent, 4), evaluate(net, data.val_x, data.val_y)[0]))
            t0 = cpu_seconds()

    net.eval()
    train_acc, train_loss = evaluate(net, data.train_x[:N_TRAIN_CHECK], data.train_y[:N_TRAIN_CHECK])
    val_acc, val_loss = evaluate(net, data.val_x, data.val_y)
    test_acc, test_loss = evaluate(net, data.test_x, data.test_y)
    if diverged:
        train_acc = val_acc = test_acc = task.chance
        train_loss = val_loss = test_loss = float("inf")
    result = {
        "task": task_name, "widths": list(widths), "activation": activation, "lr": lr, "seed": seed,
        "steps": steps, "params": count_params(task.n_in, widths, task.n_out),
        "weights": weights_per_layer(task.n_in, widths, task.n_out),
        "train_acc": train_acc, "val_acc": val_acc, "test_acc": test_acc,
        "train_loss": _num(train_loss), "val_loss": _num(val_loss), "test_loss": _num(test_loss),
        "dead": round(dead_units(net, data.test_x), 4), "seconds": round(spent, 4),
        "diverged": diverged, "curve": curve,
    }
    if extras:
        if task.grid:
            result["map"] = decision_map(net).numpy()
        if task_name == "digits1000":
            result["filters"] = net.layers[0].weight.detach()[:32].numpy().astype("float16")
    return result


N_TRAIN_CHECK = 5000


def _num(v):
    return round(v, 5) if math.isfinite(v) else None
