import copy
import random
from abc import abstractmethod
from typing import Tuple, Dict, Iterable

import numpy
from scipy.spatial.distance import cdist
# not cheating, see the scipy documentation
from scipy.stats import entropy as kl_div

from clusterlets import Clusterlet
from clusterlets.extractors.clusters import Matcher
from clusterlets.extractors.matches import Match, SoftPartition


class PinballMatcher(Matcher):
    """Matches each cluster of one label to the closest of the other label. **Requires binary labels!**"""
    def __init__(self, hops: int = 1, seed: int = 42):
        """Matches each cluster of one label to the closest of the other label. Requires binary labels!

        Args:
            hops: How many hops to perform when creating matches. With 1 hop, if Ai matches Bj, we have a match {Ai, Bj}.
                With 2 hops, if Ai matches Bj, and Bj matches Ak, then we have a match {Ai, Bj, Ak}. Each hop "jumps"
                from one label-specific clusterlet set to the other, following the clusterlet at
                minimum distance. Defaults to 1.
        """
        if hops < 1:
            raise ValueError("Hops must be a positive integer.")
        self.hops = hops
        self.seed = seed

    def __str__(self):
        return f"PinballMatcher with {self.hops} hops."

    def get_params(self) -> Dict:
        return {"matcher": self.__class__.__name__, "hops": self.hops}

    @abstractmethod
    def cost(self, these_clusterlets: list[Clusterlet], those_clusterlets: list[Clusterlet]) -> numpy.ndarray:
        """Compute a distance metric between `these_clusterlets` and `those_clusterlets`.

        Args:
            these_clusterlets: List of `n` clusterlets.
            those_clusterlets: List of `m` clusterlets.

        Returns:
            An `n` by `m` distance matrix between `these_clusterlets` and `those_clusterlets`.
        """
        pass

    def match(self, clusterlets: set[Clusterlet], **kwargs) -> SoftPartition:
        """Match the given clusterings.

        Args:
            clusterlets: The clusterlets to match.

        Returns:
            An array assigning every instance clustered to a clusterlet.
        """
        random.seed(self.seed)

        # dirty hack: next(iter(...)) peeks the set without popping
        unique_labels = numpy.arange(next(iter(clusterlets)).label_frequencies.size)
        per_label_clusterlets = [[c for c in clusterlets if numpy.argmax(c.label_frequencies).item() == label]
                                 for label in unique_labels]
        bipartite_distances = self.cost(per_label_clusterlets[0], per_label_clusterlets[1])

        # matches are bidirectional to avoid having
        left_to_right_matches = bipartite_distances.argmin(axis=1)  # column 1: left, column 2: right
        right_to_left_matches = bipartite_distances.argmin(axis=0)  # column 1: right, column 2: left
        left_to_right_matches = numpy.vstack((numpy.arange(left_to_right_matches.size),
                                             left_to_right_matches)).transpose()
        right_to_left_matches = numpy.vstack((numpy.arange(right_to_left_matches.size),
                                             right_to_left_matches)).transpose()

        matches = left_to_right_matches
        right_matching = set(matches[:, 1])
        # if left-to-right matches do not find all right clusterlets, then add them
        if right_matching != set(range(len(per_label_clusterlets[1]))):
            missing_right_clusterlets = set(range(len(per_label_clusterlets[1]))) - right_matching
            # right_to_left have inverted indices, need to swap them
            matches_to_add = right_to_left_matches[list(missing_right_clusterlets)][:, [1, 0]]

            matches = numpy.vstack((matches, matches_to_add))

        if self.hops == 1:
            # non-overlapping matches simply match one-to-one
            partition = SoftPartition({
                Match({per_label_clusterlets[0][left_match], per_label_clusterlets[1][right_match]})
                for left_match, right_match in matches
            })
        else:
            partition = list()
            for left_match, right_match in matches:
                hitlist = [left_match, right_match]  # hits of the pinball, only indicated with their _id

                is_forward_direction = False
                for hop in range(self.hops - 1):
                    player = hitlist[-1]
                    if not is_forward_direction: # left matches of a right hit: pinballing from right to left
                        hitlist.append(right_to_left_matches[player, 1])
                    else: # right matches of a left hit: pinballing from left to right
                        hitlist.append(left_to_right_matches[player, 1])

                    # update direction
                    is_forward_direction = not is_forward_direction

                # update hitlist
                block = Match({per_label_clusterlets[i % 2][hit]  # i % 2 to get the correct list
                               for i, hit in enumerate(hitlist)})
                partition.append(block)

            partition = SoftPartition(partition)


        return partition


class GreedyPinballMatcher(PinballMatcher):
    """A pinball matcher which aggregates clusterlets by greedily minimizing match cost, stopping when no
    greedy cost reduction is possible."""
    def __init__(self, max_block_size: int = 2, seed: int = 42):
        """

        Args:
            max_block_size: Maximum number of clusterlets per partition.
        """
        super(GreedyPinballMatcher, self).__init__(hops=1, seed=seed)
        self.max_block_size = max_block_size

        # memoization for storing costs between sets of clusterlets
        self._cost_memory: Dict[SoftPartition, float] = dict()

    def __str__(self):
        return f"GreedyPinballMatcher with {self.max_block_size} maximum block size."

    def get_params(self) -> Dict:
        return {"matcher": self.__class__.__name__, "max_block_size": self.max_block_size}

    def greedy_direction(self, match: Match, candidates: list[Clusterlet]) -> Tuple[int, float]:
        """Of all the given `candidates`, which one lowers the cost when aggregated with `match`?

        Args:
            match: The current match
            candidates: Possible clusterlets to add to `match`

        Returns:
            Cost and index of the best candidate.
        """
        costs = self.cost([Clusterlet.merge(match)], candidates)[0]
        best = costs.argmin()

        return best, costs[best]

    def _sample_seed(self, candidates: list[Clusterlet]) -> Clusterlet:
        """Select a random element from `candidates`."""
        # python takes O(n) for a sample from a set: BOOOOO!!1!
        return random.sample(tuple(candidates), 1)[0]

    def match(self, clusterlets: set[Clusterlet], **kwargs) -> SoftPartition:
        """Match the given clusterings.

       Args:
           clusterlets: The clusterlets to match.

       Returns:
           An array assigning every instance clustered to a clusterlet.
       """
        random.seed(self.seed)

        available_clusterlets = [copy.deepcopy(clusterlet) for clusterlet in clusterlets]

        seed = copy.deepcopy(self._sample_seed(available_clusterlets))
        del available_clusterlets[available_clusterlets.index(seed)]

        current_match = Match({seed})
        current_cost = +numpy.inf
        partition = SoftPartition()

        while len(available_clusterlets) > 0:
            # end of chain?
            if len(current_match) == self.max_block_size:
                partition.add(current_match)
                # update cost and match
                seed = copy.deepcopy(self._sample_seed(available_clusterlets))
                current_match = Match({seed})
                current_cost = +numpy.inf
                del available_clusterlets[available_clusterlets.index(seed)]

                continue

            # extend the chain?
            greedy_match_index, match_cost = self.greedy_direction(current_match, available_clusterlets)
            if match_cost <= current_cost:
                # yes
                current_match.add(copy.deepcopy(available_clusterlets[greedy_match_index]))
                current_cost = match_cost

                del available_clusterlets[greedy_match_index]
            else:
                # no
                partition.add(current_match)

                # update cost and match
                seed = copy.deepcopy(self._sample_seed(available_clusterlets))
                current_match = Match({seed})
                current_cost = +numpy.inf
                del available_clusterlets[available_clusterlets.index(seed)]

                continue

        # leftover block
        if len(current_match) > 0:
            partition.add(current_match)

        return partition

################
### Matchers ###
################
class DPbMatcher(PinballMatcher):
    """A pinball matcher based on centroid distances."""
    def __init__(self, hops: int, seed: int = 42):
        super(DPbMatcher, self).__init__(hops=hops, seed=seed)

    def __str__(self):
        return f"DistancePinballMatcher with {self.hops} hops."

    def cost(self, these_clusterlets: list[Clusterlet], those_clusterlets: list[Clusterlet]) -> numpy.ndarray:
        """Compute distances as centroid distances."""
        cumulative_frequencies = sum([clusterlet.label_frequencies
                                      for clusterlet in these_clusterlets + those_clusterlets])
        unique_labels = numpy.arange(these_clusterlets[0].label_frequencies.size)


        per_label_clusterlets = [[c for c in these_clusterlets + those_clusterlets
                                  if numpy.argmax(c.label_frequencies).item() == label]
                                 for label in unique_labels]

        # clusterlets with the same label: awful!
        if any(cumulative_frequencies == 0) or any(len(c) == 0 for c in per_label_clusterlets):
            return numpy.full(shape=(len(these_clusterlets), len(those_clusterlets)), fill_value=numpy.inf)

        centroids = [numpy.vstack([c.centroid for c in clusterlets])
                     for clusterlets in per_label_clusterlets]

        return cdist(centroids[0], centroids[1])

class BalancePbMatcher(PinballMatcher):
    """A pinball matcher based on balance."""
    def __init__(self, hops: int, target_ratio: numpy.ndarray = None, seed: int = 42):
        """
        Args:
            target_ratio: The desired ratio. Distance measured as distance from `target_ratio`
        """
        super().__init__(hops=hops, seed=seed)
        self.target_ratio = target_ratio

    def __str__(self):
        return f"BalancePinballMatcher with {self.target_ratio} target ratio and {self.hops} hops."

    def get_params(self) -> Dict:
        return {"matcher": self.__class__.__name__, "hops": self.hops, "target_ratio": self.target_ratio.tolist()}

    def distribution_distance(self, reference_distribution: numpy.ndarray, other_distribution: numpy.ndarray) -> float:
        """Measure distance between distributions.

        Args:
            reference_distribution: Distribution to compare.
            other_distribution: Distribution to compare.

        Returns:
            A distance between `this_distribution` and `that_distribution`. Measured as KL divergence.
        """
        return kl_div(reference_distribution, other_distribution)

    def clusterlets_distribution(self, clusterlets: Iterable[Clusterlet]) -> numpy.ndarray:
        distribution = sum([clusterlet.label_frequencies for clusterlet in clusterlets])
        distribution = distribution / sum(distribution)

        return distribution

    def cost(self, these_clusterlets: list[Clusterlet], those_clusterlets: list[Clusterlet]) -> numpy.ndarray:
        """Compute distances as centroid distances."""
        cost_matrix = numpy.full(shape=(len(these_clusterlets), len(those_clusterlets)), fill_value=numpy.nan)
        for i, this_clusterlet in enumerate(these_clusterlets):
            for j, that_clusterlet in enumerate(those_clusterlets):
                if j < i:
                    # lower triangle, can copy from upper
                    cost_matrix[i, j] = cost_matrix[j, i]
                else:
                    # upper triangle: gotta compute
                    cost_matrix[i, j] = self.distribution_distance(
                        reference_distribution=self.target_ratio,
                        other_distribution=self.clusterlets_distribution([this_clusterlet, that_clusterlet])
                    )

        return cost_matrix

#####################
### EarlyMatchers ###
#####################
class GreedyDPbMatcher(GreedyPinballMatcher, DPbMatcher):
    def __init__(self, max_block_size: int = 2, seed: int = 42):
        super(GreedyDPbMatcher, self).__init__(max_block_size=max_block_size, seed=seed)

    def __str__(self):
        return f"Greedy PinballMatcher with {self.max_block_size} hops."

    def get_params(self) -> Dict:
        return {"matcher": self.__class__.__name__, "max_block_size": self.max_block_size}

class GreedyBalanceMatcher(GreedyPinballMatcher, BalancePbMatcher):
    def __init__(self, max_block_size: int, target_ratio: numpy.ndarray, seed: int = 42):
        super(GreedyBalanceMatcher, self).__init__(seed=seed)
        self.max_block_size = max_block_size
        self.target_ratio = target_ratio

    def __str__(self):
        return f"Greedy PinballMatcher with {self.target_ratio} target ratio and {self.max_block_size} max block size."

    def get_params(self) -> Dict:
        return {"matcher": self.__class__.__name__, "max_block_size": self.max_block_size, "target_ratio": self.target_ratio.tolist()}
