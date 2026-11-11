"""The trainer the experiments actually use: several networks of the same shape trained side by side.

It computes exactly what netsize/train.py computes (same starting weights, same
batches, same Adam updates, same learning-rate schedule), and tests/ checks
that. It exists only because of speed. On a small network almost all of the
time goes on PyTorch's bookkeeping, not arithmetic, so training three networks
in one batched call costs little more than training one; and writing the
backward pass and Adam out by hand removes most of the rest.

Network k of the group gets learning rate lrs[k] and seed seeds[k]. The seed
decides its starting weights, its training set and the order of its batches.
"""
import math

import torch
import torch.nn.functional as F

from .clock import cpu_seconds
from .nets import MLP, count_params, layer_sizes, weights_per_layer
from .tasks import get_task
from .train import BATCH, CHECKPOINTS, GRID, GRID_SPAN, N_TRAIN_CHECK, _num, _schedule

BETA1, BETA2, EPS = 0.9, 0.999, 1e-8          # Adam's usual settings (the PyTorch defaults)
_SQRT2, _SQRT2PI = math.sqrt(2.0), math.sqrt(2.0 * math.pi)

torch.set_flush_denormal(True)


class Group:
    """K networks of one shape. All their parameters live in one K x P table, one row per network."""

    def __init__(self, n_in, widths, n_out, activation, seeds):
        nets = [MLP(n_in, widths, n_out, activation, seed=s) for s in seeds]
        self.activation = activation
        self.sizes = layer_sizes(n_in, widths, n_out)
        self.K = len(nets)
        P = count_params(n_in, widths, n_out)
        self.theta = torch.empty(self.K, P)
        self.grad = torch.zeros(self.K, P)
        self.W, self.b, self.gW, self.gb = [], [], [], []
        at = 0
        for l, (a, b) in enumerate(zip(self.sizes[:-1], self.sizes[1:])):
            for table, ws, bs in ((self.theta, self.W, self.b), (self.grad, self.gW, self.gb)):
                ws.append(table[:, at:at + a * b].view(self.K, a, b))
                bs.append(table[:, at + a * b:at + a * b + b].view(self.K, 1, b))
            at += a * b + b
            for k, net in enumerate(nets):
                self.W[l][k].copy_(net.layers[l].weight.detach().T)
                self.b[l][k, 0].copy_(net.layers[l].bias.detach())
        self.m = torch.zeros(self.K, P)
        self.v = torch.zeros(self.K, P)
        self.scratch = torch.empty(self.K, P)
        self.t = 0

    # -- forward ---------------------------------------------------------------

    def forward(self, x, keep=None):
        """x is [K, N, inputs]. Returns the outputs [K, N, classes].

        With keep (a list), also stores what the backward pass needs: for every
        layer its input, and for every hidden layer its output (and, for GELU,
        the value before the activation).
        """
        h = x
        last = len(self.W) - 1
        for l in range(last + 1):
            z = torch.baddbmm(self.b[l], h, self.W[l])
            if l == last:
                if keep is not None:
                    keep.append((h, None, None))
                return z
            if self.activation == "relu":
                a, z = z.relu_(), None
            elif self.activation == "leaky_relu":
                a, z = F.leaky_relu_(z, 0.01), None
            elif self.activation == "tanh":
                a, z = z.tanh_(), None
            else:
                a = F.gelu(z)
            if keep is not None:
                keep.append((h, a, z))
            h = a

    def _slope(self, a, z):
        """Derivative of the activation, from its output a (and its input z for GELU)."""
        if self.activation == "relu":
            return a.sign()                               # a is never negative, so this is 1 where the unit fired
        if self.activation == "leaky_relu":
            return a.sign().clamp_(min=0.01)
        if self.activation == "tanh":
            return 1.0 - a * a
        return 0.5 * (1.0 + torch.erf(z / _SQRT2)) + z * torch.exp(-0.5 * z * z) / _SQRT2PI

    # -- one training step -------------------------------------------------------

    def step(self, x, y, lr):
        """One Adam step on a batch. x [K, B, inputs], y [K, B] class numbers, lr [K, 1] learning rates."""
        keep = []
        out = self.forward(x, keep)
        # gradient of the mean cross-entropy loss with respect to the outputs
        delta = torch.softmax(out, 2)
        delta.scatter_add_(2, y.unsqueeze(2), torch.full((1, 1, 1), -1.0).expand(*y.shape, 1))
        delta.div_(x.shape[1])
        for l in range(len(self.W) - 1, -1, -1):
            h = keep[l][0]
            torch.bmm(h.transpose(1, 2), delta, out=self.gW[l])
            torch.sum(delta, 1, keepdim=True, out=self.gb[l])
            if l > 0:
                delta = torch.bmm(delta, self.W[l].transpose(1, 2))
                delta.mul_(self._slope(keep[l - 1][1], keep[l - 1][2]))
        # Adam, on the whole table at once
        self.t += 1
        g, m, v, s = self.grad, self.m, self.v, self.scratch
        m.mul_(BETA1).add_(g, alpha=1.0 - BETA1)
        v.mul_(BETA2).addcmul_(g, g, value=1.0 - BETA2)
        torch.sqrt(v, out=s)
        s.div_(math.sqrt(1.0 - BETA2 ** self.t)).add_(EPS)
        torch.div(m, s, out=s)
        self.theta.addcmul_(s, lr, value=-1.0 / (1.0 - BETA1 ** self.t))

    # -- measurements ------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, x, y, chunk=2048):
        """Accuracy and mean loss of every network. x is [N, inputs] (shared) or [K, N, inputs]."""
        if x.dim() == 2:
            x, y = x.unsqueeze(0).expand(self.K, *x.shape), y.unsqueeze(0).expand(self.K, *y.shape)
        right = torch.zeros(self.K)
        loss = torch.zeros(self.K)
        for i in range(0, x.shape[1], chunk):
            out = self.forward(x[:, i:i + chunk])
            yy = y[:, i:i + chunk]
            right += (out.argmax(2) == yy).sum(1)
            loss += F.cross_entropy(out.reshape(-1, out.shape[2]), yy.reshape(-1), reduction="none").view(self.K, -1).sum(1)
        n = x.shape[1]
        return (right / n).tolist(), (loss / n).tolist()

    @torch.no_grad()
    def dead_units(self, x, chunk=2048):
        """Per network: the fraction of hidden units that output exactly zero for every example."""
        x = x.unsqueeze(0).expand(self.K, *x.shape)
        alive = None
        for i in range(0, x.shape[1], chunk):
            keep = []
            self.forward(x[:, i:i + chunk], keep)
            fired = [(a != 0).any(1) for _, a, _ in keep[:-1]]
            alive = fired if alive is None else [p | q for p, q in zip(alive, fired)]
        total = sum(a.shape[1] for a in alive)
        if total == 0:                                    # no hidden layer at all
            return [0.0] * self.K
        return [1.0 - sum(int(a[k].sum()) for a in alive) / total for k in range(self.K)]

    @torch.no_grad()
    def decision_maps(self):
        axis = torch.linspace(-GRID_SPAN, GRID_SPAN, GRID)
        gy, gx = torch.meshgrid(axis, axis, indexing="ij")
        pts = torch.stack([gx.reshape(-1), gy.reshape(-1)], 1).unsqueeze(0).expand(self.K, -1, -1)
        p = torch.softmax(self.forward(pts), 2)[:, :, 1]
        return (p * 255).round().to(torch.uint8).reshape(self.K, GRID, GRID)


@torch.no_grad()
def train_group(task_name, widths, activation, lrs, seeds, steps=None, extras=False):
    """Train len(lrs) networks of one shape together; return one result dictionary per network."""
    assert len(lrs) == len(seeds)
    task = get_task(task_name)
    steps = int(steps or task.steps)
    K = len(lrs)
    data = [task.data(s) for s in seeds]
    same_data = len(set(seeds)) == 1
    group = Group(task.n_in, widths, task.n_out, activation, seeds)
    peak = torch.tensor(lrs, dtype=torch.float32).view(K, 1)

    n = task.n_train
    gens = [torch.Generator().manual_seed(1000 * s + 17) for s in seeds]
    orders = [torch.randperm(n, generator=g) for g in gens]
    at = 0
    marks = sorted({int(round(steps * (k + 1) / CHECKPOINTS)) for k in range(CHECKPOINTS)})
    curves = [[] for _ in range(K)]
    spent = 0.0
    t0 = cpu_seconds()
    for step in range(steps):
        if at + BATCH > n:
            orders, at = [torch.randperm(n, generator=g) for g in gens], 0
        if same_data:
            idx = orders[0][at:at + BATCH]
            x = torch.index_select(data[0].train_x, 0, idx).unsqueeze(0).expand(K, -1, -1)
            y = torch.index_select(data[0].train_y, 0, idx).unsqueeze(0).expand(K, -1)
        else:
            x = torch.stack([torch.index_select(d.train_x, 0, o[at:at + BATCH]) for d, o in zip(data, orders)])
            y = torch.stack([torch.index_select(d.train_y, 0, o[at:at + BATCH]) for d, o in zip(data, orders)])
        at += BATCH
        group.step(x, y, peak * (_schedule(step, steps, 1.0)))
        if step + 1 == marks[len(curves[0])]:
            spent += cpu_seconds() - t0             # the clock stops while the networks are being tested
            acc, _ = group.evaluate(data[0].val_x, data[0].val_y)
            for k in range(K):
                curves[k].append((step + 1, round(spent / K, 4), acc[k]))
            t0 = cpu_seconds()

    tx = torch.stack([d.train_x[:N_TRAIN_CHECK] for d in data])
    ty = torch.stack([d.train_y[:N_TRAIN_CHECK] for d in data])
    train_acc, train_loss = group.evaluate(tx, ty)
    val_acc, val_loss = group.evaluate(data[0].val_x, data[0].val_y)
    test_acc, test_loss = group.evaluate(data[0].test_x, data[0].test_y)
    dead = group.dead_units(data[0].test_x)
    maps = group.decision_maps() if extras and task.grid else None
    results = []
    for k in range(K):
        diverged = not all(math.isfinite(v) for v in (train_loss[k], val_loss[k], test_loss[k]))
        r = {
            "task": task_name, "widths": list(widths), "activation": activation, "lr": lrs[k], "seed": seeds[k],
            "steps": steps, "params": count_params(task.n_in, widths, task.n_out),
            "weights": weights_per_layer(task.n_in, widths, task.n_out),
            "train_acc": train_acc[k], "val_acc": val_acc[k], "test_acc": test_acc[k],
            "train_loss": _num(train_loss[k]), "val_loss": _num(val_loss[k]), "test_loss": _num(test_loss[k]),
            "dead": round(dead[k], 4), "seconds": round(spent / K, 4), "diverged": diverged, "curve": curves[k],
        }
        if diverged:
            r["train_acc"] = r["val_acc"] = r["test_acc"] = task.chance
        if maps is not None:
            r["map"] = maps[k].numpy()
        if extras and task_name == "digits1000":
            r["filters"] = group.W[0][k, :, :32].T.contiguous().numpy().astype("float16")
        results.append(r)
    return results
