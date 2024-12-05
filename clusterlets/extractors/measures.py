import numpy


def binary_balance(labels: numpy.ndarray) -> float:
    """Compute balance of the given `labels`. `labels` must be binary!"""
    _, counts = numpy.unique(labels, return_counts=True)
    minimum, maximum = counts.min(), counts.max()
    balance = minimum / maximum

    return balance
