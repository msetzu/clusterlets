"""
Extract clusterlets from data. A clusterlet is a very small cluster satisfying some balance property.
"""
import copy
from abc import abstractmethod
from typing import Dict

import more_itertools
import numpy

from clusterlets import Clusterlet


class ClusterletExtractor(object):
    """Extracts clusterlets from the given data."""
    @abstractmethod
    def extract(self, data: numpy.ndarray, labels: numpy.ndarray, *args, **kwargs) -> set[Clusterlet]:
        """Extract clusterlets from the given `data`, each instance assigned to a `label`,

        Args:
            data: The data
            labels: The labels
            *args:
            **kwargs:

        Returns:
            An array assigning each instance in data to a clusterlet
        """
        pass

    @abstractmethod
    def get_params(self) -> Dict:
        """Extract the parameters of the extractor."""
        pass

class RandomExtractor(ClusterletExtractor):
    """Random clusterlets balancing label proportions."""
    def __init__(self, random_state: int):
        self.random_state = random_state

    def __str__(self):
        return f"RandomExtractor with {self.random_state} random seed."

    def get_params(self) -> Dict:
        return {"extractor": self.__class__.__name__, "random_state": self.random_state}

    def extract(self, data: numpy.ndarray, labels: numpy.ndarray, *args, **kwargs) -> set[Clusterlet]:
        """Random clusterlets balancing label proportions.

        Args:
            data: The data to be decomposed.
            labels: Labels assigned to each cluster.
            size_per_label: Number/proportion of instances per label. If set to "auto", computes distribution on
                            `data`, and replicates the same distribution on each clusterlet.

        Returns:
            A labeling assigning each instance in `data` to a clusterlet.
        """
        size_per_label = kwargs.get("size_per_label", "auto")
        unique_labels, counts_per_label = numpy.unique(labels, return_counts=True)
        argsort = numpy.argsort(counts_per_label)
        unique_labels, counts_per_label = unique_labels[argsort], counts_per_label[argsort]

        indexes_per_label = {
            label: numpy.argwhere(labels == label).reshape(-1)
            for label in unique_labels
        }
        # shuffle indexes of each label so they can later be directly sampled
        for label in indexes_per_label:
            numpy.random.shuffle(indexes_per_label[label])

        # compute size per clusterlet
        if size_per_label == "auto":
            denominator = numpy.gcd(*counts_per_label)
            clusterlet_size_per_label = (counts_per_label / denominator).astype(int)
            clusterlet_size_per_label = dict(zip(unique_labels, clusterlet_size_per_label))
        else:
            clusterlet_size_per_label = copy.deepcopy(size_per_label)

        # split indexes according to determined label size per clusterlet
        clusterlet_indexes_per_label = {
            label: list(more_itertools.batched(indexes_per_label[label], n=clusterlet_size_per_label[label]))
            for label in unique_labels
        }
        number_of_clusterlets = len(clusterlet_indexes_per_label[0])
        clusterlet_indexes = [
            numpy.hstack([
                numpy.array(clusterlet_indexes_per_label[label][clusterlet_number])
                for label in unique_labels
            ])
            for clusterlet_number in range(number_of_clusterlets)
        ]

        # combine indexes, picking the random i-th sample of indexes of each label
        extracted_clusterlets = set()
        for clusterlet_number, clusterlet_index in enumerate(clusterlet_indexes):
            frequencies = numpy.zeros(shape=unique_labels.shape)
            for label, count in clusterlet_size_per_label.items():
                frequencies[int(label)] = count

            extracted_clusterlets.add(Clusterlet(
                id_=clusterlet_number,
                centroid=None,
                label_frequencies=frequencies,
                index=clusterlet_index
            ))

        return extracted_clusterlets
