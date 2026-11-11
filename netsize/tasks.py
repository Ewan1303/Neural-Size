"""The tasks: six classification problems, all generated from scratch.

They differ in two ways that matter for the size of a network:

    how many inputs there are          2, 96 or 1,000
    how complicated the rule is        from "is the point inside a circle" to
                                       "which digit is somewhere in this noisy picture"

Every task hands out a fixed training set (which depends on the seed) and a
validation set and a test set that are the same for every run, so two networks
are always scored on the same questions.
"""
import functools
import zlib
from dataclasses import dataclass, field

import numpy as np
import torch

N_VAL = 2000
N_TEST = 5000


def _rng(*key):
    """A random generator named by words and numbers, e.g. _rng("spirals", "train", 3)."""
    ints = [zlib.crc32(str(k).encode()) for k in key]
    return np.random.default_rng(ints)


# ----------------------------------------------------------------------------
# Two inputs
# ----------------------------------------------------------------------------

def circle_points(rng, n):
    """Points in a square; the class is whether the point lies inside a circle that covers half of it."""
    x = rng.uniform(-1.0, 1.0, (n, 2))
    y = (x ** 2).sum(1) < 2.0 / np.pi
    return (x * 1.7).astype(np.float32), y.astype(np.int64)


def spiral_latent(rng, n, turns, jitter=0.02):
    """Two interleaved spiral arms inside the unit disc. Returns points and which arm each is on."""
    y = rng.integers(0, 2, n)
    t = np.sqrt(rng.uniform(0.0, 1.0, n))            # square root: points spread evenly over the disc
    angle = 2.0 * np.pi * turns * t + np.pi * y
    pts = t[:, None] * np.stack([np.cos(angle), np.sin(angle)], 1)
    pts = pts + rng.normal(0.0, jitter, (n, 2))
    return pts, y.astype(np.int64)


def spiral_points(rng, n, turns):
    pts, y = spiral_latent(rng, n, turns)
    return (pts * 2.0).astype(np.float32), y


# ----------------------------------------------------------------------------
# A thousand inputs
# ----------------------------------------------------------------------------

N_BIG = 1000


def needle_points(rng, n):
    """1,000 independent noisy inputs of which only 10 matter.

    Two hidden groups of 5 inputs each; the class is the pair of signs of the two
    group sums (4 classes). The other 990 inputs are pure distraction.
    """
    where = _rng("needle", "structure").permutation(N_BIG)[:10]
    x = rng.normal(0.0, 1.0, (n, N_BIG)).astype(np.float32)
    a = x[:, where[:5]].sum(1) > 0
    b = x[:, where[5:]].sum(1) > 0
    return x, (2 * a + b).astype(np.int64)


SENSOR_NOISE = 0.2
SENSOR_SCALE = 2.0 / np.sqrt(N_BIG)


def sensor_points(rng, n, turns):
    """The spirals again, but seen through 1,000 sensors.

    Each sensor reports its own fixed mixture of the point's two coordinates plus
    a little fresh noise. All 1,000 inputs carry information, yet underneath
    there are still only two numbers. The readings are scaled so that the whole
    1,000-number input is about as long as the two-number input of the plain
    spiral task (each single reading is small).
    """
    mix = _rng("sensors", "structure").normal(0.0, 1.0, (N_BIG, 2))
    pts, y = spiral_latent(rng, n, turns)
    x = (pts @ mix.T + rng.normal(0.0, SENSOR_NOISE, (n, N_BIG))) * SENSOR_SCALE
    return x.astype(np.float32), y


# ----------------------------------------------------------------------------
# Digits
# ----------------------------------------------------------------------------

_FONT = """
.###. ..#.. .###. ####. ...#. ##### ..##. ##### .###. .###.
#...# .##.. #...# ....# ..##. #.... .#... ....# #...# #...#
#..## ..#.. ....# ....# .#.#. ####. #.... ...#. #...# #...#
#.#.# ..#.. ...#. .###. #..#. ....# ####. ..#.. .###. .####
##..# ..#.. ..#.. ....# ##### ....# #...# .#... #...# ....#
#...# ..#.. .#... ....# ...#. #...# #...# .#... #...# ...#.
.###. .###. ##### ####. ...#. .###. .###. .#... .###. .##..
"""


def _glyphs():
    rows = [r.split() for r in _FONT.strip().splitlines()]
    return np.array([[[c == "#" for c in rows[r][d]] for r in range(7)] for d in range(10)], dtype=np.float32)


GLYPHS = _glyphs()                                    # [digit, 7 rows, 5 columns]


def digit_images(rng, n, width, height, scales, noise):
    """A digit from a 5x7 font, blown up by a random whole-number scale and dropped at a
    random place on a width x height canvas, with Gaussian noise on every pixel.

    Returns the flattened pictures (roughly zero mean, unit spread) and the digit.
    """
    y = rng.integers(0, 10, n)
    scale = rng.choice(np.asarray(scales), n)
    img = np.zeros((n, height, width), dtype=np.float32)
    big = {s: np.kron(GLYPHS, np.ones((s, s), dtype=np.float32)) for s in scales}
    top = (rng.uniform(0.0, 1.0, n) * (height - 7 * scale + 1)).astype(int)
    left = (rng.uniform(0.0, 1.0, n) * (width - 5 * scale + 1)).astype(int)
    for i in range(n):
        s = scale[i]
        img[i, top[i]:top[i] + 7 * s, left[i]:left[i] + 5 * s] = big[s][y[i]]
    img += rng.normal(0.0, noise, img.shape).astype(np.float32)
    x = (img.reshape(n, -1) - 0.15) / np.sqrt(noise ** 2 + 0.13)
    return x.astype(np.float32), y.astype(np.int64)


# ----------------------------------------------------------------------------
# The task list
# ----------------------------------------------------------------------------

@dataclass
class Data:
    train_x: torch.Tensor
    train_y: torch.Tensor
    val_x: torch.Tensor
    val_y: torch.Tensor
    test_x: torch.Tensor
    test_y: torch.Tensor


@dataclass(frozen=True)
class Task:
    name: str
    title: str
    n_in: int
    n_out: int
    n_train: int
    steps: int                       # training steps every network gets on this task
    sample: object = field(repr=False, compare=False, default=None)     # (rng, n) -> inputs, classes
    grid: bool = False               # True when the two inputs can be drawn as a picture

    @property
    def chance(self):
        return 1.0 / self.n_out

    def fixed(self, split):
        return _held_out(self.name, split)

    def data(self, seed):
        tx, ty = _train_set(self.name, int(seed))
        vx, vy = _held_out(self.name, "val")
        sx, sy = _held_out(self.name, "test")
        return Data(tx, ty, vx, vy, sx, sy)


SPIRAL_TURNS = 3.0

_TASKS = {}


def _add(task):
    _TASKS[task.name] = task
    return task


_add(Task("circle", "Circle", 2, 2, 2000, 2000, circle_points, grid=True))
_add(Task("spirals", "Spirals", 2, 2, 4000, 6000, lambda rng, n: spiral_points(rng, n, SPIRAL_TURNS), grid=True))
_add(Task("digits96", "Small digits", 96, 10, 10000, 3000,
          lambda rng, n: digit_images(rng, n, 8, 12, (1,), 0.3)))
_add(Task("needle", "Needles", N_BIG, 4, 10000, 2000, needle_points))
_add(Task("sensors", "Sensors", N_BIG, 2, 10000, 4000, lambda rng, n: sensor_points(rng, n, SPIRAL_TURNS)))
_add(Task("digits1000", "Big digits", N_BIG, 10, 20000, 3000,
          lambda rng, n: digit_images(rng, n, 25, 40, (3, 4), 0.5)))

MAIN_TASKS = ("circle", "spirals", "digits96", "needle", "sensors", "digits1000")


def get_task(name):
    """A task by name. "spirals@1.5" is the spiral task with 1.5 turns instead of the standard 3."""
    if name in _TASKS:
        return _TASKS[name]
    base, _, arg = name.partition("@")
    if base == "spirals" and arg:
        turns = float(arg)
        return _add(Task(name, f"Spirals, {turns:g} turns", 2, 2, 4000, 6000,
                         lambda rng, n: spiral_points(rng, n, turns), grid=True))
    raise KeyError(name)


def _tensors(x, y):
    return torch.from_numpy(x), torch.from_numpy(y)


@functools.lru_cache(maxsize=2)
def _train_set(name, seed):
    task = get_task(name)
    return _tensors(*task.sample(_rng(name, "train", seed), task.n_train))


@functools.lru_cache(maxsize=None)
def _held_out(name, split):
    task = get_task(name)
    return _tensors(*task.sample(_rng(name, split), N_VAL if split == "val" else N_TEST))
