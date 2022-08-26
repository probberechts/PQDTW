# -*- coding: UTF-8 -*-
"""
dtaidistance.clustering.kshape
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(requires version 2.2.0 or higher)

Time series clustering using k-shape.

:author: Pieter Robberechts
:copyright: Copyright 2020-2021 KU Leuven, DTAI Research Group.
:license: Apache License, Version 2.0, see LICENSE for details.

"""
import logging
import math
import multiprocessing as mp


logger = logging.getLogger("be.kuleuven.dtai.pqdtw")

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

sbd_cc = None
try:
    from . import sbd_cc
except ImportError:
    logger.debug('DTAIDistance C library not available')
    sbd_cc = None


try:
    import numpy as np
except ImportError:
    np = None

from pqdtw.sbd import distance, align
from dtaidistance.util import SeriesContainer
from dtaidistance.clustering.medoids import Medoids

def _distance_with_params(t):
    series, avgs, dists_options = t
    min_i = -1
    min_d = float('inf')
    for i, avg in enumerate(avgs):
        d = distance(series, avg)
        if d < min_d:
            min_d = d
            min_i = i
    return min_i, min_d


def _distance_c_with_params(t):
    series, means, dists_options = t
    min_i = -1
    min_d = float('inf')
    for i, mean in enumerate(means):
        d = distance(series, mean)
        if d < min_d:
            min_d = d
            min_i = i
    return min_i, min_d


def zscore(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    mns = a.mean(axis=axis)
    sstd = a.std(axis=axis, ddof=ddof)
    if axis and mns.ndim < a.ndim:
        res = ((a - np.expand_dims(mns, axis=axis)) /
               np.expand_dims(sstd, axis=axis))
    else:
        res = (a - mns) / sstd
    return np.nan_to_num(res)


class KShape(Medoids):

    def __init__(self, k, max_it=100, show_progress=True):
        """k-Shape clustering algorithm.

        :param k: Number of components
        :param max_it: Maximal iterations
        :param show_progress:
        """
        self.max_it = max_it
        super().__init__(None, {}, k, show_progress)

    def fit_fast(self, series):
        return self.fit(series, use_c=True, use_parallel=True)

    def fit(self, series, use_c=False, use_parallel=False):
        """Perform K-shape clustering.

        :param series: Container with series
        :param use_c: Use the C-library (only available if package is compiled)
        :param use_parallel: Use multipool for parallelization
        :return: cluster indices, number of iterations
            If the number of iterations is equal to max_it, the clustering
            did not converge.
        """
        if np is None:
            raise NumpyException("Numpy is required for the KShape.fit method.")
        self.series = SeriesContainer.wrap(series, support_ndim=False)

        if use_c:
            fn = _distance_c_with_params
        else:
            fn = _distance_with_params

        indices = np.random.randint(0, self.k, size=len(self.series))
        self.centroids = np.zeros((self.k, len(self.series[0])))
        performed_it = 0

        # Iterations
        it_nbs = range(self.max_it)
        if self.show_progress and tqdm is not None:
            it_nbs = tqdm(it_nbs)
        for _ in it_nbs:
            performed_it += 1
            old_indices = indices
            # Refinement step
            for j in range(self.k):
                self._extract_shape(indices, j, use_c)

            # Assignment step
            if use_parallel:
                with mp.Pool() as p:
                    clusters_distances = p.map(fn, [(self.series[idx], self.centroids, self.dists_options) for idx in
                                                    range(len(self.series))])
            else:
                clusters_distances = list(map(fn, [(self.series[idx], self.centroids, self.dists_options) for idx in
                                                   range(len(self.series))]))
            indices = [cluster for (cluster, _) in clusters_distances]

            if np.array_equal(old_indices, indices):
                break

        self.cluster_idx = {ki: set() for ki in range(self.k)}
        for idx, cluster in enumerate(indices):
            self.cluster_idx[cluster].add(idx)
        return self.cluster_idx, performed_it

    def _extract_shape(self, idx, j, use_c):
        """ Extract the most representative shape from the underlying data. """
        _a = []
        for i in range(len(idx)):
            if idx[i] == j:
                if np.sum(self.centroids[j]) == 0:
                    opt_x = self.series[i]
                else:
                    opt_x = align(self.centroids[j], self.series[i], use_c)
                _a.append(opt_x)
        a = np.array(_a)

        if len(a) == 0:
            #TODO: ?
            return np.zeros((1, len(self.series[0])))

        columns = a.shape[1]
        y = zscore(a, axis=1, ddof=1)
        s = np.dot(y.transpose(), y)

        p = np.empty((columns, columns))
        p.fill(1.0/columns)
        p = np.eye(columns) - p

        m = np.dot(np.dot(p, s), p)
        _, vec = np.linalg.eigh(m)
        centroid = vec[:, -1]
        finddistance1 = math.sqrt(((a[0] - centroid) ** 2).sum())
        finddistance2 = math.sqrt(((a[0] + centroid) ** 2).sum())

        if finddistance1 >= finddistance2:
            centroid *= -1

        self.centroids[j] = zscore(centroid, ddof=1)


