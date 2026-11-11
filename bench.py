"""Measure what networks of different sizes cost to train and to run.

    python bench.py                  # writes results/cost.csv (a minute or two)

Run it when the computer is otherwise idle. Unlike the experiments, which train
several networks at once with a hand-written trainer, this times ordinary
PyTorch (autograd and torch.optim.Adam) on one network at a time on one core,
which is what you would get if you trained that network yourself. No learning
happens here: the inputs are random numbers, because the time a step takes does
not depend on what the numbers are.

    training step    forward pass, loss, backward pass and Adam update on a batch of 128
    one example      forward pass only, in batches of 1,000, divided by 1,000
"""
import argparse
import csv
import os

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import torch
import torch.nn.functional as F

from netsize.clock import cpu_seconds
from netsize.experiments import BUDGETS, DEPTHS, MAX_WIDTH, REST, SWEEP_REST_WIDTHS, SWEEP_WIDTHS
from netsize.nets import MLP, count_params, flat_shape
from netsize.train import BATCH


def _time(step, block=0.08, most=400, blocks=6):
    """Seconds per call of step(): the fastest of several timed blocks (so a moment of interference does not count)."""
    for _ in range(3):
        step()
    best = float("inf")
    for _ in range(blocks):
        n, t0 = 0, cpu_seconds()
        while True:
            step()
            n += 1
            spent = cpu_seconds() - t0
            if spent >= block or n >= most:
                break
        best = min(best, spent / n)
    return best


def measure(n_in, widths, n_out):
    torch.manual_seed(0)
    net = MLP(n_in, widths, n_out)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    x, y = torch.randn(BATCH, n_in), torch.randint(0, n_out, (BATCH,))

    def train_step():
        loss = F.cross_entropy(net(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    big = torch.randn(1000, n_in)

    def infer():
        with torch.no_grad():
            net(big)

    return _time(train_step) * 1e3, _time(infer) / 1000 * 1e6


def shapes():
    out = []
    for n_in, n_out in ((2, 2), (96, 10), (1000, 10)):
        for depth in DEPTHS:
            for budget in BUDGETS[n_in]:
                w = flat_shape(n_in, n_out, depth, budget)
                if w is not None and max(w) <= MAX_WIDTH:
                    out.append(("flat", n_in, n_out, w))
    for budget in (10000, 100000):                       # one enormous layer against three ordinary ones
        out.append(("wide", 2, 2, flat_shape(2, 2, 1, budget)))
        out.append(("wide", 2, 2, flat_shape(2, 2, 3, budget)))
    for h in SWEEP_WIDTHS:
        out.append(("sweep_first", 1000, 10, (h, REST, REST)))
    for h in SWEEP_REST_WIDTHS:
        out.append(("sweep_rest", 1000, 10, (REST, h, h)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    torch.set_num_threads(1)
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for kind, n_in, n_out, widths in shapes():
        train_ms, infer_us = measure(n_in, widths, n_out)
        rows.append(dict(kind=kind, n_in=n_in, n_out=n_out, depth=len(widths), widths="-".join(map(str, widths)),
                         params=count_params(n_in, widths, n_out), train_ms=round(train_ms, 4), infer_us=round(infer_us, 4)))
        print(f"{kind:12s} {n_in:5d} inputs  {rows[-1]['widths']:38s} {rows[-1]['params']:>8} parameters"
              f"  step {train_ms:8.3f} ms   one example {infer_us:8.3f} us", flush=True)
    with open(os.path.join(a.out, "cost.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", os.path.join(a.out, "cost.csv"))


if __name__ == "__main__":
    main()
