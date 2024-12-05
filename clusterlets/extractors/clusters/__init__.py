"""
Extract clusterlets by clustering on each label, then matching label-specific clusterlets.
"""
import copy
from abc import abstractmethod
from typing import Optional, Dict

import more_itertools
import numpy
from sklearn.cluster import KMeans

from clusterlets import Clusterlet
from clusterlets.extractors import ClusterletExtractor
from clusterlets.extractors.matches import Matcher


class ClusteringExtractor(ClusterletExtractor):
    def __init__(self, matcher: Optional[Matcher] = None, **hyperparameters):
        self.hyperparameters = dict(hyperparameters)
        self._distances = None
        self.target_ratio = None
        self.matcher = matcher

    @abstractmethod
    def cluster(self, data: numpy.ndarray, indexes: numpy.ndarray, label_set: numpy.ndarray, label: int) -> set[Clusterlet]:
        """Cluster the given data into a clusterlet.

        Args:
            data: The data to cluster
            indexes: Original indexes of the data: used to map the given data (with index [0, 1, ...]) to the
            original data (with arbitrary index). Use when the given data is a subset/scramble of the original one.
            label_set: A range of labels.
            label: The label of the given data.
        """
        pass

    def extract(self, data: numpy.ndarray, labels: numpy.ndarray, *args, **kwargs) -> set[Clusterlet]:
        """First, cluster each label, then match clusters with `self.matcher`.

        Args:
            data: The data to be decomposed.
            labels: Labels assigned to each cluster.
        """
        unique_labels, counts = numpy.unique(labels, return_counts=True)
        unique_labels, counts = unique_labels.astype(int), counts.astype(int)
        target_ratio = counts / labels.size
        indexes_per_label = [numpy.argwhere(labels == label).reshape(-1) for label in unique_labels]

        # cluster each label separately
        clusterlets = [self.cluster(data[label_indexes], label_indexes, label_set=unique_labels, label=label)
                       for label, label_indexes in zip(unique_labels, indexes_per_label)]
        for label_clusterlets, label in zip(clusterlets, unique_labels):
            for clusterlet in label_clusterlets:
                frequencies = numpy.full(shape=unique_labels.shape, fill_value=0, dtype=numpy.int64)
                # can recover size by the index since only one label is present
                frequencies[label.item()] = clusterlet.index.size
                clusterlet.label_frequencies = frequencies

        base_clusterlets = set(more_itertools.flatten(clusterlets))
        # match each label-specific cluster to other clusters
        clusterlets = self.matcher.match(base_clusterlets, target_ratio=target_ratio)
        clusterlets = set(Clusterlet.merge(c) for c in clusterlets)

        # assign random ids
        for i, clusterlet in enumerate(clusterlets):
            clusterlet.id_ = i

        return clusterlets


class KMeansExtractor(ClusteringExtractor):
    def __init__(self, matcher: Optional[Matcher] = None, clustering_hyperparameters: Dict = None):
        super().__init__(matcher=matcher)
        self.clustering_hyperparameters = clustering_hyperparameters if clustering_hyperparameters is not None else dict()
        self.clustering_algorithm = KMeans(n_clusters=self.clustering_hyperparameters.get("k", 10),
                                           init="k-means++")

    def __str__(self):
        return f"KMeansExtractor with {self.matcher} matcher."

    def get_params(self) -> Dict:
        d = {
            "extractor": self.__class__.__name__,
            "clustering_algorithm": self.clustering_algorithm.__class__.__name__,
        }
        d.update(self.clustering_hyperparameters)

        return d

    def cluster(self, data: numpy.ndarray, indexes: numpy.ndarray, label_set: numpy.ndarray, label: int) -> set[Clusterlet]:
        algorithm = copy.deepcopy(self.clustering_algorithm)
        algorithm.fit(data)

        clusterlets = Clusterlet.from_centroid_clustering(algorithm, label_set=label_set, label=label)
        for clusterlet in clusterlets:
            clusterlet.index = indexes[clusterlet.index]

        return clusterlets
