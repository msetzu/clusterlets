# Clusfairing

Clusfairing is a Python library collecting algorithms for fair clustering, mainly aimed at *clusterlet*-based
approaches. A *clusterlet* defines (like [fair coresets](http://arxiv.org/abs/1812.10854) and [fairlets](https://proceedings.neurips.cc/paper/2017/hash/978fce5bcc4eccc88ad48ce3914124a2-Abstract.html))
a clustering of data which respects some notion of fairness or balance.
Under some assumptions, (centroids of)
clusterlets can be clustered themselves achieve a fair clustering where each cluster approximately follows the original
label distribution.

---

# Quickstart

### Installation
```shell
mkvirtualenv -p python3.12 clusfairing
pip install -r requirements.txt
```

### Getting started

```python
import numpy

from clusterlets.extractors import RandomExtractor


# generate some data
data = numpy.random.rand(1000, 5)
labels = numpy.random.choice([0, 1], 1000, replace=True)

# creates a clusterlet extractor
extractor = RandomExtractor(random_state=42)
# extracts clusterlets, assigning a clusterlet to each data point
extracted_clusterlets = extractor.extract(data, labels, size_per_label="auto")

# you can access the data of each clusterlet!
for clusterlet in extracted_clusterlets:
    print(data[clusterlet.index])
```

**Clusterlets.** The `Clusterlet` class implements a clusterlet, which is defined by
- `_id: int` an id to identify it
- `label_frequecies: Optional[numpy.ndarray]` label frequencies associated to id
- `centroid: Optional[numpy.ndarray]` a centroid
- `index: Optional[numpy.ndarray]` indicating which instances of the starting data compose this clusterlet. Used in place of the data itself for a lighter object 

Since clusterlets are extracted from a dataset, `data[clusterlet.index]` yields the data of the clusterlet.
Clusterlets support `==` and `hash`, thus can be aggregated into `set[Clusterlet]`, and used as dictionary keys.


**Extractors.** Clusterlets are extracted with a set of extractors implementing the `ClusterletExtractor` interface
(`extractors.*`),
which extracts clusterlets through the `extract(data, labels)` method.
- `RandomExtractor` selects random subsets of each label, then pairs them to satisfy dataset balance. One can also specify how many samples per label each clusterlet must have with a parameter dictionary `size_per_label`
- `KMeansExtractor` clusters each label separately (through K-Means), creating label-specific clusterlets. Then, matches clusterlets to achieve both clustering and balance.

The `KMeansExtractor` is an implementation of the `ClusteringExtractor` interface, which can be adapted to any
clustering algorithm by overriding the `cluster(data)` method.

**Matchers.** Matchers (`extractors.matches.*`) are objects which "match" existing clusterlets, creating larger ones,
i.e., they cluster clusterlets.
A `Matcher` implements a `match(clusterlets, **kwargs)` method, which is given a list of clusterings (one per
label), and a desired label balance to achieve.
Currently, we implement:
- `PinballMatcher`, which provides matches by hopping `hops` times through two sets of clusterlets of different labels, each hop following the clusterlet of opposite label at minimum distance.
- `GreedyPinballMatcher`, which greedily matches clusterlets maximizing some given objective:
  - `GreedyBalanceMatcher` maximizes label balance
  - `GreedyDPbMatcher` maximizes clusterlet distance
- `CentroidMatcher`, which creates a set of candidate partitions of the set of clusterlets, then scores them for balance and compactness. Note: only a subsample of size `sample_size` is tested due to the [superexponential number of possible partitions](https://en.wikipedia.org/wiki/Bell_number).