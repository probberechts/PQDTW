# -*- coding: UTF-8 -*-
#cython: wraparound=False
#cython: boundscheck=False
#cython: nonecheck=False

"""
dtaidistance.dtw_search
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dynamic Time Warping (DTW) Nearest Neighbor Search

This module implements nearest neighbor search classification in time series
data using the LB_Keogh algorithm as a lower bounding measure to speed up the
DTW distance computations.

:author: Pieter Robberechts, Wannes Meert, Kenneth Devloo
:copyright: Copyright 2020 KU Leuven, DTAI Research Group.
:license: Apache License, Version 2.0, see LICENSE for details.

"""
import array
import heapq
import itertools
import logging
import math
import random
from collections import Counter

from dtaidistance import util_numpy
from dtaidistance.dtw import distance
from dtaidistance.util import SeriesContainer

logger = logging.getLogger("be.kuleuven.dtai.pqdtw")

dtw_search_cc = None
try:
    from . import dtw_search_cc
except ImportError:
    logger.debug('DTAIDistance C library not available')
    dtw_search_cc = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    if util_numpy.test_without_numpy():
        raise ImportError()
    import numpy as np
    list_types = (list, array.array, set, tuple, np.ndarray)
    array_min = np.min
    array_max = np.max
except ImportError:
    np = None
    list_types = (list, array.array, set, tuple)
    array_min = min
    array_max = max


def _check_library(raise_exception=True):
    if dtw_search_cc is None:
        msg = "The compiled dtaidistance C library is not available.\n" + \
              "See the documentation for alternative installation options."
        logger.error(msg)
        if raise_exception:
            raise Exception(msg)


def __lb_keogh_envelope(s, window, use_c):
    """Compute envelope of a single series."""
    sz = len(s)
    window = sz if window is None else window

    if use_c:
        l, u = dtw_search_cc.lb_keogh_envelope(s, window=window)
    else:
        l = array.array('d', [0]*sz)
        u = array.array('d', [0]*sz)
        for i in range(sz):
            imin = max(0, i - window)
            imax = min(i + window + 1, sz)
            l[i] = array_min(s[imin:imax])
            u[i] = array_max(s[imin:imax])
    return l, u


def lb_keogh_envelope_fast(data, window=None):
    """Compute time-series envelope as required by LB_Keogh.

    .. seealso:: :method:`lb_keogh_envelope`
    """
    return lb_keogh_envelope(data, window, True, True)


def lb_keogh_envelope(s, window=None, parallel=False, use_c=False, use_mp=False):
    """Compute time-series envelope as required by LB_Keogh.

    :param s: A series or iterable of series for which the envelope should be computed.
    :param window: window to be used for the envelope generation (the envelope at time
        index i will be generated based on all observations from the time
        series at indices comprised between i-window and i+window). If None, the
        observations from the entire series are used.
    :param parallel: Use parallel operations
    :param use_c: Use c compiled Python functions
    :param use_mp: Use Multiprocessing for parallel operations (not OpenMP)
    :returns: A tuple or list tuples containg the lower- and uppper-side of
    the envelope of each series in s.

    .. seealso:: :method:`lb_keogh`: Compute LB_Keogh similarity

    :References:

    .. [1] Keogh, E. Exact indexing of dynamic time warping. In International
       Conference on Very Large Data Bases, 2002. pp 406-417.
    """

    # Check whether multiprocessing is available
    if use_c:
        _check_library(raise_exception=True)
    if parallel and (use_mp or not use_c):
        try:
            import multiprocessing as mp
            logger.info('Using multiprocessing')
        except ImportError:
            msg = 'Cannot load multiprocessing'
            logger.error(msg)
            raise Exception(msg)
    else:
        mp = None

    # If s is not an iterable
    if not isinstance(s[0], list_types):
        logger.info('Computing envelope of s')
        return __lb_keogh_envelope(s, window, use_c)

    logger.info('Computing envelope of each series in s')
    s = SeriesContainer.wrap(s)
    d = len(s)

    lang = 'C' if use_c else 'Python'
    if use_c and parallel and not use_mp:
        logger.info("Compute envelopes in C (parallel=OMP)")
        envelopes = dtw_search_cc.lb_keogh_envelope_parallel(s, window=window)
    elif parallel and use_mp:
        logger.info("Compute envelope in {} (parallel=MP)".format(lang))
        with mp.Pool() as p:
            envelopes = p.map(__lb_keogh_envelope, [(s[di], use_c, {'window' : window}) for di in range(d)])
    else:
        logger.info("Compute envelopes in {} (parallel=No)".format(lang))
        envelopes = []
        for di in range(d):
            l, u = __lb_keogh_envelope(s[di], window, use_c)
            envelopes.append((l, u))

    return envelopes


def __lb_keogh(s1, s2, s2_envelope, window, use_c):
    """Compute LB Keogh of a single series."""
    if s2_envelope is not None:
        if use_c:
            return dtw_search_cc.lb_keogh_from_envelope(s1, s2_envelope)

        l, u = s2_envelope
        t = 0
        for i in range(len(s1)):
            ci = s1[i]
            if (ci > u[i]):
                t += (ci - u[i]) ** 2
            elif (ci < l[i]):
                t += (l[i] - ci) ** 2
        return math.sqrt(t)

    else:
        l1 = len(s1)
        l2 = len(s2)
        window = max(l1, l2) if window is None else window
        if use_c:
            return dtw_search_cc.lb_keogh(s1, s2, window=window)

        lb = 0
        ldiff12 = l1 + 1;
        if (ldiff12 > l2):
            ldiff12 -= l2;
            if (ldiff12 > window):
                ldiff12 -= window
            else:
                ldiff12 = 0
        else:
            ldiff12 = 0
        ldiff21 = l2 + window
        if (ldiff21 > l1):
            ldiff21 -= l1
        else:
            ldiff21 = 0

        for i in range(l1):
            if (i > ldiff12):
                imin = i - ldiff12
            else:
                imin = 0
            imax = max(l2, ldiff21)
            ui = array_max(s2[imin:imax])
            li = array_min(s2[imin:imax])
            ci = s1[i]
            if (ci > ui):
                lb += (ci - ui) ** 2
            elif (ci < li):
                lb += (li - ci) ** 2
        return math.sqrt(lb)


def lb_keogh_fast(s1, s2=None, s2_envelope=None, window=None):
    """Compute LB_Keogh distance.

    .. seealso:: :method:`lb_keogh`
    """
    return lb_keogh(s1, s2, s2_envelope, window, True, True)


def lb_keogh(s1, s2=None, s2_envelope=None, window=None, parallel=False, use_c=False, use_mp=False):
    """Compute LB_Keogh.

    :param s1: A series or iterable of series to compare to the envelope(s) of s2.
    :param s2: A series or iterable of candidate series. None means the
        envelope is provided via `s2_envelope` parameter and hence does not
        need to be computed again.
    :param s2_envelope: Pre-computed envelope(s) of the candidate time series.
        If set to None, it is computed based on `s2`.
    :param window: Window to be used for the envelope generation (the envelope
        at time index i will be generated based on all observations from the
        candidate series at indices comprised between i-window and i+window).
        Not used if `s2` is None.
    :param parallel: Use parallel operations
    :param use_c: Use c compiled Python functions
    :param use_mp: Use Multiprocessing for parallel operations (not OpenMP)
    :returns: Distance between the series in s1 and the envelope of the
        candidate series in s2.

    .. seealso:: :method:`lb_envelope`: Compute LB_Keogh-related envelope

    :References:

    .. [1] Keogh, E. Exact indexing of dynamic time warping. In International
       Conference on Very Large Data Bases, 2002. pp 406-417.
    """
    # Check whether multiprocessing is available
    if use_c:
        _check_library(raise_exception=True)
    if parallel and (use_mp or not use_c):
        try:
            import multiprocessing as mp
            logger.info('Using multiprocessing')
        except ImportError:
            msg = 'Cannot load multiprocessing'
            logger.error(msg)
            raise Exception(msg)
    else:
        mp = None

    # Check whether s1 and s2 are iterators and
    # whether all series are of equal length
    if isinstance(s1[0], list_types):
        l1 = [len(s) for s in s1]
        s1_iterator = True
    else:
        l1 = [len(s1)]
        s1_iterator = False
    if s2 is not None and isinstance(s2[0], list_types):
        l2 = [len(s) for s in s2]
        s2_iterator = True
    elif s2 is not None:
        l2 = [len(s2)]
        s2_iterator = False
    elif s2_envelope is not None and isinstance(s2_envelope[0][0], list_types):
        l2 = [len(l) for (l, _) in s2_envelope]
        s2_iterator = True
    else:
        l2 = [len(s2_envelope[0])]
        s2_iterator = False
    equal_series = all(l == l1[0] for l in l1 + l2)

    # Precompute the envelopes if all input series are equal in length
    if s2_envelope is None and equal_series:
        logger.info('Precomputing LB Keogh envelope for each series in s2')
        s2_envelope = lb_keogh_envelope(s2, window, parallel, use_c)
    elif s2_envelope is not None and not equal_series:
        raise ValueError("If you provide an envelope, all series should be equal in lenght.")

    # Both s1 and s2(_envelope) are a single series
    if (not s1_iterator and not s2_iterator):
        return __lb_keogh(s1, s2, s2_envelope, window, use_c)

    # Make user s1, s2 and s2_envelope are iterators
    if not s1_iterator:
        s1 = [s1]
    if s2 is None:
        s2 = [None] * len(l2)
    elif not s2_iterator:
        s2 = [s2]
    if s2_envelope is None:
        s2_envelope = [None] * len(l2)
    elif not s2_iterator:
        s2_envelope = [s2_envelope]

    # Compute distance between each pair of timeseries
    s1 = SeriesContainer.wrap(np.asarray(s1))
    s2 = SeriesContainer.wrap(np.asarray(s2))   # FIXME: only compatible with DTWSeriesMatrix for now
    d = len(s1)

    lang = 'C' if use_c else 'Python'
    if use_c and parallel and not use_mp:
        logger.info("Compute LB Keogh distances in C (parallel=OMP)")
        if equal_series:
            lb = dtw_search_cc.lb_keogh_from_envelope_parallel(s1, s2_envelope)
        else:
            lb = dtw_search_cc.lb_keogh_parallel(s1, s2, window=window)
    elif parallel and use_mp:
        logger.info("Compute LB Keogh distances in {} (parallel=MP)".format(lang))
        with mp.Pool() as p:
            lb = p.starmap(__lb_keogh, [(s1[di], s2[ei], s2_envelope[ei], window, use_c) for (di, ei) in itertools.product(range(len(s1)), range(len(s2_envelope)))])
            lb = np.array(lb).reshape((len(s1), len(s2)))
    else:
        logger.info("Compute LB Keogh distances in {} (parallel=No)".format(lang))
        lb = np.zeros((d, len(l2)))
        for di in range(d):
            for ei in range(len(l2)):
                    lb[di, ei] = __lb_keogh(s1[di], s2[ei], s2_envelope[ei], window, use_c)

    return lb


def __nearest_neighbor_lb_keogh(s, t, env, dist_params, use_c):
    """Compute nearest neighbor with LB Keogh pruning for a single query series."""
    if use_c:
        loc, _ = dtw_search_cc.nearest_neighbour_lb_keogh(1, s, t, env, **dist_params)
    else:
        loc = 0
        best_score_so_far = np.inf
        for di in range(len(s)):
            lb = lb_keogh(s[di], t, env, dist_params.get('window', None), parallel=False, use_c=False, use_mp=False)
            if best_score_so_far > lb:
                score = distance(s[di], t, max_dist=best_score_so_far, **dist_params)
                if score < best_score_so_far:
                    best_score_so_far = score
                    loc = di
    return loc


def nearest_neighbour_lb_keogh_fast(data, target, t_envelope, dist_params={}):
    """LB Keogh nearest neighbour (1NN)

    .. seealso:: :method:`nearest_neighbour_lb_keogh`
    """
    return nearest_neighbour_lb_keogh(data, target, t_envelope, dist_params, True, True)


def nearest_neighbour_lb_keogh(s, t=None, t_envelope=None, dist_params={}, use_c=False, parallel=False, use_mp=False):
    """LB Keogh nearest neighbour (1NN)

    :param s: 2D Array of 1D series or 1D series for subsequence search.
    :param t: A series or iterable of query series. None means the
        envelope is provided via `t_envelope` parameter and hence does not
        need to be computed again.
    :param t_envelope: Pre-computed envelope(s) of the query series.
        If set to None, it is computed based on `t`.
    :param dist_params: Distance function paraneters. For correctness, 'window' should match the envelope window.
    :param use_c: Use fast pure c compiled functions
    :param parallel: Use fast parallel version (only in C version)
    :param use_mp: Use Multiprocessing for parallel operations (not OpenMP)
    :returns: 1D array of nearest neighbours indices from `t` or `t_envelope`.

    :References:

    .. [1] Rakthanmanon, T. et al. Searching and Mining Trillions of Time
     Series Subsequences under Dynamic Time Warping. In proc. of SIGKDD, 2012.
    """

    if use_c:
        if dtw_search_cc is None:
            logger.warning("C-library not available, using the Python version")
            use_c = False
    if parallel and (use_mp or not use_c):
        try:
            import multiprocessing as mp
            logger.info('Using multiprocessing')
        except ImportError:
            msg = 'Cannot load multiprocessing'
            logger.error(msg)
            raise Exception(msg)
    else:
        mp = None

    logger.info("Starting nearest neighbors search")

    s = SeriesContainer.wrap(s)

    if t_envelope is None:
        t_envelope = lb_keogh_envelope(t, dist_params.get('window', None), parallel, use_c, use_mp)

    # There is only one query
    if not isinstance(t[0], list_types):
        return __nearest_neighbor_lb_keogh(s, t, t_envelope, dist_params, use_c)

    t = SeriesContainer.wrap(t)
    d = len(t)

    lang = 'C' if use_c else 'Python'
    if use_c and parallel and not use_mp:
        logger.info("Compute nearest neighbors in C (parallel=OMP)")
        locations, _ = dtw_search_cc.nearest_neighbour_lb_keogh_parallel(1, s, t, t_envelope, **dist_params)
        return locations
    elif parallel and use_mp:
        logger.info("Compute nearest neighbors in {} (parallel=MP)".format(lang))
        with mp.Pool() as p:
            return p.starmap(__nearest_neighbor_lb_keogh, [(s, t[ti], t_envelope[ti], dist_params, use_c) for ti in range(d)])
    else:
        logger.info("Compute nearest neighbors in {} (parallel=No)".format(lang))
        best_fits = array.array('i', [0]*d)
        for ti in range(d):
            best_fits[ti] = __nearest_neighbor_lb_keogh(s, t[ti, :], t_envelope[ti], dist_params, use_c)
        return best_fits

class DTWKNN:
    """A k-Nearest Neighbor classifier that compares differing length
    observation sequences using the efficient FastDTW dynamic time warping
    algorithm.

    :param k: Number of neighbors.
    :type k: int
    :param metric: Callable to compute the distance between two time series.
    :type metric: callable
    :param use_lb: Use LB Keogh for early stopping.
    :type use_lb: bool
    :param window: Window used for lower bound.
    :type window: int
    """

    def __init__(self, k, metric=None, use_lb=True, window=None):
        self.k = k
        if metric is None:
            self._metric = lambda s1, s2, max_dist: distance(s1, s2, max_dist=max_dist, use_c=True, window=window)
        else:
            self._metric = metric
        self.use_lb = use_lb
        self.window = window

    def fit(self, X, y=None, parallel=False):
        """Fits the classifier by adding labeled training observation sequences.

        :param X: A list of multiple observation sequences.
        :type X: List[numpy.ndarray]
        :param y: A list of labels for the observation sequences.
        :type y: List[str]
        """
        self._X, self._y = X, y

        if self.use_lb:
            self._envelope = lb_keogh_envelope(X, window=self.window, parallel=parallel, use_c=True, use_mp=False)

    def query(self, X, show_progress=True, parallel=False, use_c=False):
        try:
            (self._X)
        except AttributeError:
            raise RuntimeError('The classifier needs to be fitted before a query can be made')

        # Compute LB Keogh
        if self.use_lb:
            lb = lb_keogh(X, self._X, self._envelope, self.window, parallel=parallel, use_c=True, use_mp=False)

        if use_c:
            return dtw_search_cc.knn(self.k, self._metric, self._X, X, lb)

        if not parallel:
            pruned = 0
            it = range(len(X))
            if show_progress:
                it = tqdm(it, desc='Classifying examples')
            results = []
            for i in it:
                best_so_far = [(-float('inf'), -1)] * self.k
                for j, _ in enumerate(self._X):
                    if not self.use_lb or -best_so_far[0][0] > lb[i,j]:
                        score = self._metric(X[i], self._X[j], max_dist=-best_so_far[0][0])
                        if score < -best_so_far[0][0]:
                            heapq.heapreplace(best_so_far, (-score, j))
                    else:
                        pruned += 1
                results.append([int(bsf[1]) for bsf in best_so_far])
            print('pruned', pruned)
            return results

        else:
            from joblib import Parallel, delayed
            from multiprocessing import cpu_count
            def parallel_predict(process, X_chunk, lb_chunk):
                results = []
                it = range(len(X_chunk))
                if show_progress:
                    it = tqdm(it, desc='Classifying examples (process {})'.format(process), position=process-1)
                for i in it:
                    best_so_far = [(-float('inf'), -1)] * self.k
                    for j in range(len(self._X)):
                        if not self.use_lb or -best_so_far[0][0] > lb_chunk[i,j]:
                            score = self._metric(X_chunk[i], self._X[j], max_dist=-best_so_far[0][0])
                            if score < -best_so_far[0][0]:
                                heapq.heapreplace(best_so_far, (-score, j))
                    results.append([bsf[1] for bsf in best_so_far])
                return results

            n_jobs = cpu_count()
            X_chunks = np.array_split(X, n_jobs)
            if self.use_lb:
                lb_chunks = np.array_split(lb, n_jobs)
            else:
                lb_chunks = [None] * n_jobs
            results = Parallel(n_jobs=n_jobs)(delayed(parallel_predict)(i+1, X_chunks[i], lb_chunks[i]) for i in range(n_jobs))
            return [nb for sublist in results for nb in sublist] # Flatten the resulting array

    def predict(self, X, show_progress=True, parallel=False):
        """Predicts the label for an observation sequence (or multiple sequences).

        :param X: An individual observation sequence or a list of multiple observation sequences.
        :type X: numpy.ndarray or List[numpy.ndarray]
        :param show_progress: Whether to display a progress bar or not.
        :type show_progress: bool
        :param n_jobs: The number of jobs to run in parallel. Setting this to -1 will use all available CPU cores.
        :type n_jobs: int
        :returns prediction(s): The predicted label(s) for the observation sequence(s).
        :type prediction(s): str or List[str]
        """
        try:
            (self._X, self._y)
        except AttributeError:
            raise RuntimeError('The classifier needs to be fitted before predictions are made')

        def find_modes(idx):
            neighbor_labels = [self._y[i] for i in idx]
            # Find the modal labels
            counter = Counter(neighbor_labels)
            max_count = max(counter.values())
            return [k for k, v in counter.items() if v == max_count]

        neighbors = self.query(X, show_progress, parallel)
        labels = [random.choice(find_modes(nb)) for nb in neighbors]
        return labels

    def evaluate(self, X, y, labels=None, show_progress=True, parallel=False):
        """Evaluates the performance of the classifier on a batch of observation sequences and their labels.

        :param X: An individual observation sequence or a list of multiple observation sequences.
        :type X: numpy.ndarray or List[numpy.ndarray]
        :param y: A list of labels for the observation sequences.
        :type y: numpy.ndarray or List[str]
        :param labels: A list of labels for ordering the axes of the confusion matrix.
        :type labels: numpy.ndarray or List[str]
        :param show_progress: Whether to display a progress bar or not.
        :type show_progress: bool
        :param n_jobs: The number of jobs to run in parallel. Setting this to -1 will use all available CPU cores.
        :type n_jobs: int
        :returns error: The categorical error rate of the classifier on the observation sequences.
        :type accuracy: float
        :returns confusion: The confusion matrix representing the discrepancy between predicted and actual labels.
        :type confusion: numpy.ndarray
        """
        from sklearn.metrics import confusion_matrix

        # Classify each observation sequence and calculate confusion matrix
        predictions = self.predict(X, show_progress=show_progress, parallel=parallel)
        cm = confusion_matrix(y, predictions, labels=labels)

        return 1 - np.sum(np.diag(cm)) / np.sum(cm), cm


class PQDTWKNN(DTWKNN):
    """A k-Nearest Neighbor classifier that compares differing length
    observation sequences using the efficient FastDTW dynamic time warping
    algorithm.

    :param k: Number of neighbors.
    :type k: int
    :param metric: Callable to compute the distance between two time series.
    :type metric: callable
    :param use_lb: Use LB Keogh for early stopping.
    :type use_lb: bool
    :param window: Window used for lower bound.
    :type window: int
    """

    def __init__(self, k, pq, use_lb=True):
        super().__init__(k, use_lb=use_lb, window=pq.subspace_size)
        self._pq = pq


    def fit(self, X, y=None, parallel=False):
        """Fits the classifier by adding labeled training observation sequences.

        :param X: A list of multiple observation sequences.
        :type X: List[numpy.ndarray]
        :param y: A list of labels for the observation sequences.
        :type y: List[str]
        """
        # self._pq.fit(X)
        self._X_encoding = self._pq.encode(X, parallel=parallel)
        super().fit(X, y, parallel)

    def query(self, X, show_progress=True, parallel=False, use_c=False):
        try:
            (self._X)
        except AttributeError:
            raise RuntimeError('The classifier needs to be fitted before predictions are made')

        X_encoding = self._pq.encode(X, parallel=parallel)

        # Compute LB Keogh
        if self.use_lb:
            lb = lb_keogh(X, self._X, self._envelope, self.window, parallel=parallel, use_c=True, use_mp=False)


        if use_c:
            return dtw_search_cc.pqknn(self.k, self._pq.approx_distance, self._X, X, self._X_encoding, X_encoding, lb)

        pruned = 0
        it = range(len(X))
        if show_progress:
            it = tqdm(it, desc='Classifying examples')
        results = []
        for i in it:
            best_so_far = [(-float('inf'), -1)] * self.k
            for j, _ in enumerate(self._X):
                if not self.use_lb or -best_so_far[0][0] > lb[i,j]:
                    score = self._pq.approx_distance(X[i,:], self._X[j,:], X_encoding[i,:], self._X_encoding[j,:])
                    if score < -best_so_far[0][0]:
                        heapq.heapreplace(best_so_far, (-score, j))
                else:
                    pruned += 1
            results.append([int(bsf[1]) for bsf in best_so_far])
        print('pruned', pruned)
        return results
