"""Checks on the pieces the results rest on. Run with: python -m pytest"""
import math
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from netsize import experiments as ex
from netsize.fast import train_group
from netsize.nets import ACTIVATIONS, MLP, count_params, first_layer_shape, first_layer_share, flat_shape, scaled_shape, weights_per_layer
from netsize.tasks import MAIN_TASKS, N_BIG, N_TEST, N_VAL, _rng, get_task
from netsize.train import train


# -- counting ---------------------------------------------------------------

@pytest.mark.parametrize("n_in,widths,n_out", [(2, (), 2), (2, (8,), 2), (96, (20, 10), 10), (1000, (64, 3, 17), 4)])
def test_parameter_count_matches_the_real_network(n_in, widths, n_out):
    net = MLP(n_in, widths, n_out)
    assert count_params(n_in, widths, n_out) == sum(p.numel() for p in net.parameters())
    assert sum(weights_per_layer(n_in, widths, n_out)) == sum(l.weight.numel() for l in net.layers)


def test_first_layer_share():
    assert first_layer_share(1000, (64, 64, 64), 10) == pytest.approx(64000 / (64000 + 4096 + 4096 + 640))
    assert first_layer_share(2, (64, 64, 64), 2) < 0.02


# -- shapes -----------------------------------------------------------------

@pytest.mark.parametrize("n_in", [2, 96, 1000])
def test_flat_shapes_hit_their_budget(n_in):
    n_out = 2 if n_in == 2 else 10
    for depth in ex.DEPTHS:
        for budget in ex.BUDGETS[n_in]:
            w = flat_shape(n_in, n_out, depth, budget)
            if w is None:
                continue
            assert len(w) == depth and len(set(w)) == 1
            assert abs(math.log(count_params(n_in, w, n_out) / budget)) <= math.log(1.25)


def test_named_shapes_keep_their_proportions():
    funnel = scaled_shape(96, 10, (4, 2, 1), 30000)
    assert funnel[0] > funnel[1] > funnel[2]
    hourglass = scaled_shape(96, 10, (4, 1, 4), 30000)
    assert hourglass[0] == hourglass[2] > hourglass[1]
    assert scaled_shape(1000, 10, (1, 1, 1), 500) is None          # cannot be done: one unit already costs 1,001


def test_first_layer_shape_spends_the_rest_on_later_layers():
    small = first_layer_shape(1000, 10, 4, 100000)
    large = first_layer_shape(1000, 10, 64, 100000)
    assert small[0] == 4 and large[0] == 64
    assert small[1] > large[1]
    for w in (small, large):
        assert abs(count_params(1000, w, 10) - 100000) / 100000 < 0.05
    assert first_layer_shape(1000, 10, 200, 100000) is None        # 200 first-layer units alone cost twice the budget


# -- tasks ------------------------------------------------------------------

@pytest.mark.parametrize("name", MAIN_TASKS + ("spirals@1.5",))
def test_tasks_are_well_formed_and_repeatable(name):
    t = get_task(name)
    a, b, other = t.data(0), t.data(0), t.data(1)
    assert a.train_x.shape == (t.n_train, t.n_in) and a.train_x.dtype == torch.float32
    assert a.val_x.shape == (N_VAL, t.n_in) and a.test_x.shape == (N_TEST, t.n_in)
    assert torch.equal(a.train_x, b.train_x) and torch.equal(a.train_y, b.train_y)
    assert not torch.equal(a.train_x, other.train_x)               # another seed, another training set
    assert torch.equal(a.test_x, other.test_x)                     # ... but the same test
    assert not torch.equal(a.val_x[:100], a.test_x[:100])
    counts = torch.bincount(a.test_y, minlength=t.n_out).float() / N_TEST
    assert counts.min() > 0.6 / t.n_out and int(a.test_y.max()) == t.n_out - 1
    assert torch.isfinite(a.train_x).all()


def test_needle_answer_depends_on_ten_inputs_only():
    x, y = get_task("needle").fixed("test")
    where = _rng("needle", "structure").permutation(N_BIG)[:10]
    x = x.numpy()
    again = 2 * (x[:, where[:5]].sum(1) > 0) + (x[:, where[5:]].sum(1) > 0)
    assert np.array_equal(again, y.numpy())


def test_sensor_readings_have_two_numbers_underneath():
    x, _ = get_task("sensors").fixed("test")
    s = torch.linalg.svdvals(x[:1000] - x[:1000].mean(0))
    assert s[1] > 5 * s[2]                                         # two big directions, then only noise


# -- the two trainers agree ---------------------------------------------------

@pytest.mark.parametrize("activation", list(ACTIVATIONS))
def test_fast_trainer_matches_plain_pytorch(activation):
    lrs, seeds = [0.003, 0.01, 0.003], [0, 0, 2]
    fast = train_group("digits96", (12, 7), activation, lrs, seeds, steps=40)
    for lr, seed, got in zip(lrs, seeds, fast):
        ref = train("digits96", (12, 7), activation, lr, seed, steps=40)
        for k in ("train_acc", "val_acc", "test_acc", "dead"):
            assert got[k] == pytest.approx(ref[k], abs=1e-6), k
        for k in ("train_loss", "val_loss", "test_loss"):
            assert got[k] == pytest.approx(ref[k], abs=2e-4), k


def test_fast_trainer_handles_no_hidden_layer_and_learns():
    r = train_group("circle", (16, 16), "relu", [0.01], [0], steps=600, extras=True)[0]
    assert r["test_acc"] > 0.95 and r["map"].shape == (64, 64)
    linear = train_group("needle", (), "relu", [0.01], [0], steps=200)[0]
    assert linear["test_acc"] > 0.6 and linear["dead"] == 0.0


# -- the experiment list ------------------------------------------------------

def test_experiment_list_is_consistent():
    entries = ex.all_entries()
    jobs = ex.unique_networks(entries)
    assert len({j["key"] for j in jobs}) == len(jobs) <= len(entries)
    for e in entries:
        t = get_task(e["task"])
        assert e["params"] == count_params(t.n_in, e["widths"], t.n_out)
        assert max(e["widths"], default=1) <= ex.MAX_WIDTH
    assert ex.lr_grid(100) == (0.003, 0.01, 0.03) and ex.lr_grid(30000) == (0.001, 0.003, 0.01)
    for e in ex.first_budget():
        assert abs(e["params"] - e["budget"]) / e["budget"] < 0.12 and e["widths"][0] == e["first"]


def test_learning_rate_search_extends_past_the_ends_only():
    assert ex.next_lr([0.003, 0.01, 0.03], 0.01) is None            # best in the middle: stop
    assert ex.next_lr([0.003, 0.01, 0.03], 0.03) == 0.1             # best at the top: try the next one up
    assert ex.next_lr([0.003, 0.01, 0.03], 0.003) == 0.001          # best at the bottom: the next one down
    assert ex.next_lr([0.003, 0.01, 0.03, 0.1], 0.1) is None        # the ladder has run out
    assert ex.next_lr([0.003, 0.01, 0.03, 0.1], 0.03) is None
    runs = [dict(role="search", lr=lr, val_acc=acc, val_loss=1.0) for lr, acc in ((0.003, 0.5), (0.01, 0.6), (0.03, 0.7))]
    assert ex.search_is_open(runs) and not ex.search_is_open(runs + [dict(role="search", lr=0.1, val_acc=0.2, val_loss=None)])
