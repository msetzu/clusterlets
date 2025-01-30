"""Clusterlets"""
from __future__ import annotations

import dataclasses
import functools
import operator
from typing import Optional, Iterable, Any

import numpy
from sklearn.cluster import KMeans


@dataclasses.dataclass
class Clusterlet:
    """A clusterlet, a cluster of instances, possibly associated to a label."""
    id_: Optional[Any]
    label_frequencies: Optional[numpy.ndarray] = None  # is clusterlet associated to a label?
    centroid: Optional[numpy.ndarray] = None  # does the clusterlet has a centroid?
    index: Optional[numpy.ndarray] = None  # indices mapping the clusterlet data to the data generating it

    def __hash__(self):
        return hash(self.id_)

    def __eq__(self, other):
        return isinstance(other, Clusterlet)\
            and self.id_ == other.id_\
            and self.index.size == other.index.size\
            and (self.index == other.index).all()\
            and (self.centroid == other.centroid).all()\
            and (self.label_frequencies == other.label_frequencies).all()

    def __str__(self):
        return f"""Clusterlet
            _id: {self.id_}
            size: {self.index.size if self.index is not None else None}
            centroid: {self.centroid if self.centroid is not None else None}
            frequencies: {self.label_frequencies}
            """

    def asdict(self):
        d = dataclasses.asdict(self)
        d["label_frequencies"] = self.label_frequencies.tolist() if self.label_frequencies is not None else None
        d["index"] = self.index.tolist()
        d["centroid"] = self.centroid.tolist() if self.centroid is not None else None

        return d

    @staticmethod
    def from_centroid_clustering(clustering: KMeans, label_set: numpy.ndarray, label: Optional[int] = None) -> set[Clusterlet]:
        """Construct a list of clusterlets from the given kmeans instance."""
        clusterlets = set()
        for i, centroid in enumerate(clustering.cluster_centers_):
            index = numpy.argwhere(clustering.labels_ == i).reshape(-1)
            label_frequencies = label_set.copy()
            label_frequencies[label] = index.size

            clusterlets.add(Clusterlet(
                id_=i,
                label_frequencies=label_frequencies,
                centroid=centroid,
                index=index
            ))

        return clusterlets

    @staticmethod
    def merge(clusterlets: Iterable[Clusterlet], remove_overlap: bool = True) -> Clusterlet:
        """

        Args:
            clusterlets: The clusterlets to merge.
            remove_overlap: True to remove overlapping indexes, False otherwise. Defaults to False.

        Returns:
            A clusterlet merge of all the others. Centroid and label given as the mean of the respective clusterlets,
            index as the union, with or without overlap (see remove_overlap). Index set to None.
        """
        if remove_overlap:
            return functools.reduce(operator.add, clusterlets)
        else:
            return functools.reduce(Clusterlet.__index_only_add, clusterlets)

    def __add__(self, other: Clusterlet) -> Clusterlet:
        """Sum two clusterlets"""
        if self.label_frequencies is not None and other.label_frequencies is not None:
            label = self.label_frequencies + other.label_frequencies
        else:
            label = None
        
        if self.centroid is not None and other.centroid is not None:
            centroid = (self.centroid + other.centroid) / 2
        else:
            centroid = None
        
        if self.index is not None and other.index is not None:
            index = numpy.hstack((self.index, other.index))
        else:
            index = None

        return Clusterlet(
            id_=None,
            label_frequencies=label,
            centroid=centroid,
            index=index
        )

    @staticmethod
    def __index_only_add(this: Clusterlet, other: Clusterlet) -> Clusterlet:
        return Clusterlet(
            id_=None,
            label_frequencies=None,
            centroid=None,
            index=numpy.hstack((this.index, other.index))
        )
