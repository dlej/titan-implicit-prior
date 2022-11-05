"""Training utilities.
"""


def normalize(x, fullnormalize=False):
    """Normalize input to lie between 0, 1.

        Args:
            x: Input signal
            fullnormalize: If True, normalize such that minimum is 0 and
                maximum is 1. Else, normalize such that maximum is 1 alone.

        Outputs:
            xnormalized: Normalized x.
    """
    if x.sum() == 0:
        return x
    xmax = x.max()
    if fullnormalize:
        xmin = x.min()
    else:
        xmin = 0
    return (x - xmin) / (xmax - xmin)
