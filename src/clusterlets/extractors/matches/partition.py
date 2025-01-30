"""Construct clusterlets by joining existing (label-specific) clusterlets together."""
import random
from typing import List, Dict

import numpy
from scipy.spatial.distance import squareform, pdist
from sklearn.metrics import silhouette_score

from clusterlets.extractors import Clusterlet
from clusterlets.extractors.matches import Matcher, SoftPartition, Match


def is_acceptable_partition(partition: List[List[Clusterlet]]) -> bool:
    """Filter-out undesired cluster matches."""
    # remove unitary partitions
    if len(partition) == 1:
        return False

    # remove partitions with missing labels in any block
    for block in partition:
        if len(block) == 1:
            return False

        block_label_frequencies = sum([clusterlet.label_frequencies for clusterlet in block])
        if any(block_label_frequencies == 0):
            return False

    return True


class CentroidMatcher(Matcher):
    """Matches clusters of different labels together to compose clusterlets. Weighs in centroid distance and
    label distribution."""
    def __init__(self, data: numpy.ndarray, labels: numpy.ndarray, sample_size: int = 250000,
                 distance_weight: float = 0.5, random_state: int = 42):
        self.data = data
        self.labels = labels
        self._distances = squareform(pdist(data))
        self.sample_size = sample_size
        self.distance_weight = distance_weight
        self.seed = random_state

    def __str__(self):
        return f"CentroidMatcher with {self.sample_size} sample size and {self.distance_weight} distance weight."

    def score_partition(self, partition: List[List[Clusterlet]],
                        clusterlet_distances: numpy.ndarray,
                        base_clusterlets: list[Clusterlet],
                        labels: numpy.ndarray,
                        target_ratio: numpy.ndarray) -> float:
        """Scores the given partition.

        Args:
            partition: The partition to score: a list of list of clusterlets
            clusterlet_distances: Distances among centroids
            base_clusterlets: Clusterlets in the same order of the centroid distances
            labels: The data labels, used to evaluate the goodness of the partition
            target_ratio: Global label ratio

        Returns:
            The partition score, the higher, the better.
        """
        if not is_acceptable_partition(partition):
            return -numpy.inf

        # label ratio
        partition_as_clusterlets = [Clusterlet.merge(block, remove_overlap=False) for block in partition]
        blocks_ratios = [numpy.unique(labels[block.index], return_counts=True)[1] / block.index.size
                         for block in partition_as_clusterlets]

        # per-block difference between block ratio and target ratio, only take maximum
        maximum_ratio_difference = abs(blocks_ratios - target_ratio).max()
        ratio_score = 1 - maximum_ratio_difference

        # centroid distance
        partition_labels = numpy.full(fill_value=numpy.nan, shape=clusterlet_distances.shape[0])
        for block_id, block_clusterlets in enumerate(partition):
            for clusterlet in block_clusterlets:
                partition_labels[base_clusterlets.index(clusterlet)] = block_id

        distance_score = silhouette_score(
            X=clusterlet_distances,
            labels=partition_labels.astype(int),
            metric="precomputed"
        )

        return self.distance_weight * distance_score + (1 - self.distance_weight) * ratio_score

    def get_params(self) -> Dict:
        return {
            "matcher": self.__class__.__name__,
            "sample_size":  self.sample_size,
            "distance_weight": self.distance_weight,
            "random_state": self.seed
        }

    def sample_partitions(self, clusterlets: List[Clusterlet], k: int = 1000) -> List[List[List[Clusterlet]]]:
        """Generate a set of `k` random partitions from the given `clusterlets`.

        Args:
            clusterlets: The clusterlets set
            k: Number of partitions to generate

        Returns:
            A k-long generator of partitions
        """
        partitions = list()
        number_of_elements = len(clusterlets)
        # balance weights to avoid blocks with only one clusterlet: the larger the block gravity, the smaller the
        # expected block size
        # todo: may want to parametrize in future extensions
        block_gravity = number_of_elements // 5

        # first block
        balancing_weights = [(block_gravity - 1) / number_of_elements
                             for _ in range(number_of_elements // (block_gravity - 1))]
        leftover_probability_mass = 1 - sum(balancing_weights)
        leftover_elements = number_of_elements - len(balancing_weights)
        balancing_weights += [leftover_probability_mass / leftover_elements] * leftover_elements
        assignments = [random.choices(range(number_of_elements), k=number_of_elements, weights=balancing_weights)
                       for _ in range(k)]
        for assignment in assignments:
            blocks_names = set(assignment)
            partitions.append([
                [
                    clusterlets[i]  # pick all clusterlets with given block
                    for i, clusterlet in enumerate(clusterlets) if assignment[i] == block
                ]
                for block in blocks_names
            ])

        return partitions

    def match(self, clusterlets: set[Clusterlet], target_ratio: numpy.ndarray, **kwargs) -> SoftPartition:
        """Match the given clusterings, creating clusters with distribution similar to `target_ratio`.

        Args:
            clusterlets: A list of label-specific clusterings. Element i should be a clustering instance for label i.
            target_ratio: Desired label ratio for the resulting clustering.

        Returns:
            The clustering, assigning each clusterlet in the `clusterings` to joined clusterlets.
        """
        random.seed(self.seed)

        # sorting clusterlets makes it easier to map indices
        sorted_clusterlets = sorted(clusterlets, key=lambda c: (numpy.argmax(c.label_frequencies).item(), c.id_))
        centroids = numpy.vstack([c.centroid for c in sorted_clusterlets])
        centroids_distances = squareform(pdist(centroids))

        # extract centroids
        candidate_matches = self.sample_partitions(clusterlets=list(clusterlets), k=self.sample_size)
        scores = numpy.array([self.score_partition(p,
                                                   clusterlet_distances=centroids_distances,
                                                   base_clusterlets=sorted_clusterlets,
                                                   labels=self.labels,
                                                   target_ratio=target_ratio)
                              for p in candidate_matches])
        best_match_idx = numpy.argmax(scores).item()
        best_match = candidate_matches[best_match_idx]
        best_match = SoftPartition(Match(set(block)) for block in best_match)

        return best_match
