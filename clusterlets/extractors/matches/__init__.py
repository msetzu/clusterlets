"""
Match (cluster) clusterlets together.
"""
from abc import abstractmethod
from typing import Dict

import numpy

from clusterlets import Clusterlet


class Match(set[Clusterlet]):
    """A match: a set of clusterlets"""
    pass

    def __hash__(self):
        return hash("".join([str(hash(clusterlet)) for clusterlet in self]))


class SoftPartition(set[Match]):
    """A soft partition of matches: a set of matches with possible overlaps."""
    def __hash__(self):
        return hash("".join([str(hash(block)) for block in self]))


class Matcher:
    """Match clusterlets together, creating larger ones. It \"clusters\" the clusterlets."""
    @abstractmethod
    def match(self, clusterlets: set[Clusterlet], target_ratio: numpy.ndarray, **kwargs) -> SoftPartition:
        """Match the given clusterings, creating clusters with distribution similar to `target_ratio`.

        Args:
            clusterlets: A list of per-label clusterlets. Element `i` should be a list of clusterlets for label `i`.
            target_ratio: Desired label ratio for the resulting clustering.

        Returns:
            The clustering, assigning each clusterlet in the `clusterings` to joined clusterlets.
        """
        pass

    @abstractmethod
    def get_params(self) -> Dict:
        """Extract the parameters of the matcher."""
        pass
