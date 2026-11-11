"""Feed-forward networks, and the arithmetic of how many weights they have.

A network here is described by one thing: the list of its hidden-layer widths.
[64, 64] is two hidden layers of 64 units; [256, 16, 16] is a wide first layer
followed by two narrow ones. Everything else (inputs, outputs, activation) is
given separately.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

# name -> (function, gain used to scale the starting weights)
ACTIVATIONS = {
    "relu": (torch.relu, math.sqrt(2.0)),
    "leaky_relu": (lambda x: F.leaky_relu(x, 0.01), math.sqrt(2.0 / (1.0 + 0.01 ** 2))),
    "gelu": (F.gelu, math.sqrt(2.0)),
    "tanh": (torch.tanh, 5.0 / 3.0),
}


def layer_sizes(n_in, widths, n_out):
    return [n_in, *widths, n_out]


def weights_per_layer(n_in, widths, n_out):
    """Connection weights in each layer, first layer first (biases not counted).

    For a feed-forward network this is also the number of multiplications one
    forward pass costs, layer by layer.
    """
    s = layer_sizes(n_in, widths, n_out)
    return [a * b for a, b in zip(s[:-1], s[1:])]


def count_params(n_in, widths, n_out):
    """Everything that is trained: weights plus biases."""
    s = layer_sizes(n_in, widths, n_out)
    return sum(a * b + b for a, b in zip(s[:-1], s[1:]))


def first_layer_share(n_in, widths, n_out):
    """Fraction of all connection weights that sit between the inputs and the first hidden layer."""
    w = weights_per_layer(n_in, widths, n_out)
    return w[0] / sum(w)


class MLP(nn.Module):
    """A plain feed-forward network: Linear, activation, Linear, activation, ..., Linear.

    Starting weights are drawn from a normal distribution with standard deviation
    gain / sqrt(fan_in) (He initialisation for the ReLU family), so the signal
    keeps roughly the same size from layer to layer whatever the widths are.
    Hidden biases start as small random numbers, the output layer at gain 1 with
    zero bias.
    """

    def __init__(self, n_in, widths, n_out, activation="relu", seed=0):
        super().__init__()
        self.act, gain = ACTIVATIONS[activation]
        self.activation = activation
        self.widths = tuple(int(w) for w in widths)
        g = torch.Generator().manual_seed(int(seed))
        sizes = layer_sizes(n_in, self.widths, n_out)
        self.layers = nn.ModuleList()
        for k, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            lin = nn.Linear(a, b)
            last = k == len(sizes) - 2
            bound = 1.0 / math.sqrt(a)
            with torch.no_grad():
                lin.weight.normal_(0.0, (1.0 if last else gain) * bound, generator=g)
                if last:
                    lin.bias.zero_()
                else:
                    lin.bias.uniform_(-bound, bound, generator=g)
            self.layers.append(lin)

    def forward(self, x, hidden=None):
        """Run the network. Pass a list as `hidden` to collect every hidden layer's output."""
        for lin in self.layers[:-1]:
            x = self.act(lin(x))
            if hidden is not None:
                hidden.append(x)
        return self.layers[-1](x)


# ----------------------------------------------------------------------------
# Shapes: turning "this many parameters, laid out like this" into layer widths
# ----------------------------------------------------------------------------

def _closest(candidates, n_in, n_out, budget, tolerance):
    """Of several width lists, the one whose parameter count is nearest the budget (None if none is near)."""
    best, best_miss = None, None
    for widths in candidates:
        miss = abs(math.log(count_params(n_in, widths, n_out) / budget))
        if best is None or miss < best_miss:
            best, best_miss = widths, miss
    if best is None or best_miss > math.log(1.0 + tolerance):
        return None
    return tuple(best)


def scaled_shape(n_in, n_out, ratios, budget, tolerance=0.25, min_width=2):
    """Hidden widths in the given proportions, scaled so the network has about `budget` parameters.

    ratios (1, 1, 1) is three equal layers, (4, 2, 1) a funnel that halves each
    layer, (1, 0.25, 1) an hourglass. Returns None when no whole-number widths
    come within `tolerance` of the budget (tiny budgets) or a layer would be
    narrower than `min_width`.
    """
    seen, candidates = set(), []
    for k in range(0, 16 * 24):                      # scale from 1 up to 2**24 in fine steps
        s = 2.0 ** (k / 16.0)
        widths = tuple(max(1, int(round(r * s))) for r in ratios)
        if widths in seen:
            continue
        seen.add(widths)
        if min(widths) < min_width:
            continue
        if count_params(n_in, widths, n_out) > budget * (1.0 + tolerance):
            break
        candidates.append(widths)
    return _closest(candidates, n_in, n_out, budget, tolerance)


def flat_shape(n_in, n_out, depth, budget, **kw):
    """`depth` hidden layers of equal width with about `budget` parameters."""
    return scaled_shape(n_in, n_out, (1.0,) * depth, budget, **kw)


def first_layer_shape(n_in, n_out, first, budget, rest_depth=2, tolerance=0.25, min_width=2):
    """[first, w, w]: a chosen first hidden layer, with the rest of the budget spent on equal later layers."""
    candidates = []
    w = min_width
    while True:
        widths = (first,) + (w,) * rest_depth
        if count_params(n_in, widths, n_out) > budget * (1.0 + tolerance):
            break
        candidates.append(widths)
        w += 1
    return _closest(candidates, n_in, n_out, budget, tolerance)
