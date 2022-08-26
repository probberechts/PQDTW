#cython: wraparound=False
#cython: boundscheck=False
#cython: nonecheck=False
#cython: initializedcheck=False

"""
dtaidistance.quantizer
~~~~~~~~~~~~~~~~~~~~~~

Approximate Dynamic Time Warping (DTW) using Product Quantization, C implementation.

:author: Pieter Robberechts
:copyright: Copyright 2021 KU Leuven, DTAI Research Group.
:license: Apache License, Version 2.0, see LICENSE for details.

"""
import array

from cpython cimport array
from cython.view cimport array as cvarray

import logging
import math

from libc.math cimport sqrt

import random
import warnings
from enum import Enum

import numpy as np
from cython import Py_ssize_t
from cython.parallel import prange
from dtaidistance.clustering import KMeans
from dtaidistance.dtw import (_distance_matrix_length, distance,
                              distance_matrix, distances_array_to_matrix)
from dtaidistance.util import SeriesContainer

from pqdtw.dtw_search import (lb_keogh, lb_keogh_envelope,
                              nearest_neighbour_lb_keogh)
from pqdtw.kshape import KShape, zscore
from pqdtw.sbd import distance_matrix as sbd_distance_matrix

cimport cython
cimport dtaidistancec_dtw
cimport dtaidistancec_dtw_search
cimport dtaidistancec_quantizer
cimport dtaidistancec_sbd
cimport numpy as np
from dtaidistance.dtw_cc cimport DTWSettings

logger = logging.getLogger("be.kuleuven.dtai.pqdtw")


try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


try:
    # DTW implmenentation of tslearn is very slow --> replace it by
    # dtaidistance's implemntation
    def kmeans_metric_fun(dataset1, dataset2=None, global_constraint=None,
                  sakoe_chiba_radius=None, itakura_max_slope=None, n_jobs=None,
                  verbose=0):
        if dataset2 is None:
            return distance_matrix(dataset1, window=sakoe_chiba_radius+1 if sakoe_chiba_radius is not None else None)
        else:
            out = np.array(distance_matrix(np.concatenate((dataset1[:,:,0], dataset2[:,:,0])),
                    block=((0,dataset1.shape[0]), (dataset1.shape[0],dataset1.shape[0]+dataset2.shape[0])),
                    window=sakoe_chiba_radius+1 if sakoe_chiba_radius is not None else None,
                    compact=True))
            return out.reshape(dataset1.shape[0], dataset2.shape[0])
    import tslearn.metrics
    tslearn.metrics.cdist_dtw = kmeans_metric_fun
    from tslearn.clustering import TimeSeriesKMeans as tslearn_KMeans
    from tslearn.utils import to_time_series_dataset
except ImportError:
    tslearn_KMeans = None


inf = float("inf")

cpdef enum Metric:
    DTW = 1
    SBD = 2

cpdef enum SubspaceType:
    NO_OVERLAP = 1
    DOUBLE_OVERLAP=2
    PRE_ALIGN=3

cpdef enum DISTANCECALCULATION:
    SYMMETRIC=0  #DEFAULT CASE. ONLY USE PRECOMPUTED DISTANCES
    ASYMMETRIC=1
    EXACT_WHEM_0DIST=2
    ASYMMETRIC_KEOGH_WHEM_0DIST=3


def as_dict(settings):
    return {
        'window' : settings.window,
        'max_dist' : settings.max_dist,
        'max_step' : settings.max_step,
        'max_length_diff' : settings.max_length_diff,
        'penalty' : settings.penalty,
        'psi' : settings.psi,
        'use_pruning' : settings.use_pruning,
        'only_ub' : settings.only_ub,
    }


def pad_for_subspace_size(data, subspace_size):
    M = math.ceil(data.shape[1]/subspace_size)
    padl = math.floor((subspace_size * M - data.shape[1]) / 2)
    padr = math.ceil((subspace_size * M - data.shape[1]) / 2)
    return np.pad(data, ((0,0),(padl, padr)), 'constant')


def interpolate_for_subspace_size(data, subspace_size):
    M = math.ceil(data.shape[1]/subspace_size)
    idx = np.arange(data.shape[1])
    new_idx = np.linspace(0, data.shape[1], subspace_size * M)
    return np.apply_along_axis(lambda d: np.interp(new_idx, idx, d), 1, data)


cdef class PQ():
    """Product Quantization (PQ) for fast approximate DTW distance estimation of time series.

    A `D`-dim input series is divided into `M` `D`/`M`-dim sub-series.
    Each sub-series is quantized into a small integer via `codebook_size` codewords.
    For the querying phase, given a new `D`-dim query series, the distance beween the query
    and the database PQ-codes are efficiently approximated via
        - Symmetric Distance.
        - Asymmetric Distance.

    All series must be np.ndarray with np.float32

    :param subspace_size: The size of a sub-series
    :param subspace_type: Whether to use overlapping or non-overlapping sub-series.
    :param codebook_size: The number of codewords for each subspace
            (typically 256, so that each sub-vector is quantized
            into 256 bits = 1 byte = uint8)
    :param dist_strategy: Strategy used for approximate distance calculation.
    :param compute_dist_correction: Whether or not to correct the estimated
            distances by a factor estimated on the training set.
    :param dist_params: Parameters passed to
            :func:`~dtaidistance.dtw.distance` for computing the approximate distance.
    :param quantization_dist_params: Parameters passed to
            :func:`~dtaidistance.dtw.distance` for finding the nearest neighor time
            series in the training set.
    :param kmeans_dist_params: Parameters passed to :func:`~dtaidistance.clustering.KMeans`

    """
    cdef readonly int M  # Number of subspaces
    cdef readonly int subspace_size # Default size of each subspace
    cdef readonly int subspace_type # Method used to construct the subspaces
    cdef readonly int metric # Metric used to compute distances between subspaces
    cdef readonly int codebook_size # Number of centroids
    cdef PQDistanceStrategy _distance_strategy # Strategy used to compute the approximate disances.

    cdef bint compute_dist_correction # Whether to correct a bias on the distance estimations
    cdef double distance_ratio # The bias correction ratio

    # Additional variables per subspace type
    ## subspace_type=DOUBLE_OVERLAP
    cdef readonly int overlap # Number of overlapping samples between two subspaces
    ## subspace_type=PRE_ALIGN
    cdef readonly int tail # Number of samples a subspace boundary is allowd to shift left
    cdef readonly int wavelet_level # Level of the MODWT transform
    ## subspace_type=DOUBLE_OVERLAP & PRE_ALIGN
    cdef readonly double overlap_corrector # The overlap correction ratio

    cdef readonly DTWSettings kmeans_dist_params # DTW settings for the k-means clustering part
    cdef readonly DTWSettings quantization_dist_params # DTW settings for the encoding of data (i.e., NN search)
    cdef readonly DTWSettings dist_params # DTW settings for the approximate distance computations

    # Internal data structures
    cdef readonly double[:,:,::1] codebook # The codebook with all subspace centroids (M x codebook_size x subspace_size)
    cdef double[:,:,::1] codebookEnvL # The upper part of the Keogh envelope of each centroid (M x codebook_size x subspace_size)
    cdef double[:,:,::1] codebookEnvU # The lower part of the Keogh envelope of each centroid (M x codebook_size x subspace_size)


    def __init__(self, subspace_size, subspace_type=SubspaceType.NO_OVERLAP,
            tail=None, wavelet_level=1, codebook_size=256, dist_strategy=None,
            metric=Metric.DTW, compute_dist_correction=False, dist_params={},
            quantization_dist_params={}, kmeans_dist_params={}):
        self.subspace_size = subspace_size
        self.subspace_type = subspace_type
        self.codebook_size = codebook_size
        self.metric = metric
        self.distance_ratio = 1.0
        self.dist_params = DTWSettings(**dist_params)
        self.quantization_dist_params = DTWSettings(**quantization_dist_params)
        self.kmeans_dist_params = DTWSettings(**kmeans_dist_params)
        self._distance_strategy = dist_strategy
        self.compute_dist_correction = compute_dist_correction
        if subspace_type != SubspaceType.PRE_ALIGN:
            self.tail = 0
        elif tail is None:
            self.tail = int(0.2 * self.subspace_size)
        else:
            self.tail = tail
        self.wavelet_level = wavelet_level

    @property
    def dist_strategy(self) -> PQDistanceStrategy:
        return self._distance_strategy

    @dist_strategy.setter
    def dist_strategy(self, strategy: PQDistanceStrategy) -> None:
        self._distance_strategy = strategy
        strategy.post_fit(self, False)
        if self.compute_dist_correction:
            warnings.warn("You have to refit this quantizer to compute the correct distance correction factor.")

    cpdef double[::1] _prepare_data_1D(self, double[::1] data):
        if self.subspace_type != SubspaceType.PRE_ALIGN:
            return data

        cdef double[::1] data_aligned = np.empty(self.M * (self.subspace_size + self.tail))
        dtaidistancec_quantizer.get_wavelet_splits(&data[0], data.shape[0], self.wavelet_level, self.subspace_size, self.tail, &data_aligned[0])
        return data_aligned

    cpdef double[:,::1] _prepare_data_2D(self, double[:,::1] data, bint parallel):
        if self.subspace_type != SubspaceType.PRE_ALIGN:
            return data

        cdef int i
        cdef int N = data.shape[0]
        cdef int D = data.shape[1]
        cdef double[:,::1] data_aligned = np.zeros((N, self.M*(self.subspace_size + self.tail)), dtype=np.double)
        if parallel:
            for i in prange(N, nogil=True):
                dtaidistancec_quantizer.get_wavelet_splits(&data[i,0], D, self.wavelet_level, self.subspace_size, self.tail, &data_aligned[i,0])
        else:
            for i in range(N):
                dtaidistancec_quantizer.get_wavelet_splits(&data[i,0], D, self.wavelet_level, self.subspace_size, self.tail, &data_aligned[i,0])
        return data_aligned


    cpdef fit(self, double[:,::1] data, int max_iter=20, int dba_max_iter=10,
        int drop_stddev=3, bint use_tslearn=False, bint use_kshape=False, 
        int seed=123, bint parallel=False):
        """Given training time series, run k-means for each sub-space and create
        codewords for each sub-space.

        This function should be run once first of all.

        :param data (np.ndarray): Training time series with shape=(N, D) and dtype=np.float32.
        :param max_iter (int): The max number of iterations for k-means. With max_iter < 1
            the result of K-means++ initialization is used.
        :param dba_max_iter (int):
        :param drop_stddev (int):
        :param use_tslearn (bool):
        :param seed (int): The seed for random process
        :returns object: self
        """
        np.random.seed(seed)
        random.seed(seed)

        cdef int N, D, m, k, rs, re

        N = data.shape[0]
        D = data.shape[1]

        if D % self.subspace_size != 0:
            raise ValueError("""
                The provided sequences cannot be split in equal subsequences
                of length {}. Adapt the subspace size or rescale the provided
                data first by running 'pad_for_subspace_size' or
                'interpolate_for_subspace_size'.
            """.format(self.subspace_size))

        # Determine the number of sub-spaces and the overlap between sub-spaces
        if self.subspace_type == SubspaceType.NO_OVERLAP:
            self.M = D / self.subspace_size
            self.overlap = 0
            self.overlap_corrector = 1.0
        elif self.subspace_type == SubspaceType.DOUBLE_OVERLAP:
            self.M = D / self.subspace_size * 2 - 1
            self.overlap = self.subspace_size / 2
            self.overlap_corrector = (D / self.subspace_size) / float(self.M)
        elif self.subspace_type == SubspaceType.PRE_ALIGN:
            self.M = D / self.subspace_size
            self.overlap = 0
            self.overlap_corrector = self.subspace_size / float(self.subspace_size + self.tail)
        else:
            raise ValueError('{} is not a valid type.'.format(self.subspace_type))

        cdef double[:,::1] data_aligned = self._prepare_data_2D(data, parallel)

        self.codebook = np.zeros((self.M, self.codebook_size, self.subspace_size + self.tail), dtype=np.double)
        if self.metric == Metric.DTW:
            self.codebookEnvL = np.empty((self.M, self.codebook_size, self.subspace_size + self.tail), dtype=np.double)
            self.codebookEnvU = np.empty((self.M, self.codebook_size, self.subspace_size + self.tail), dtype=np.double)

        for m in range(self.M):
            logger.info("Training the subspace: {} / {}".format(m+1, self.M))
            rs = m * (self.subspace_size + self.tail - self.overlap)
            re = rs + self.tail + self.subspace_size

            if N < self.codebook_size:
                logger.warning("The requested codebook size ({}) is smaller than the number of training examples ({}). No clustering is performed.".format(self.codebook_size, N))
                self.codebook.base[m,:N,:] = data_aligned[:,rs:re]

            elif max_iter < 0:
                logger.debug("Using randomly selected training sequences as centroids.")
                indices = np.random.choice(range(N), self.codebook_size, replace=False)
                self.codebook.base[m,:,:] = [data_aligned[random.randint(0, N - 1), rs:re] for _ki in indices]

            elif max_iter == 0:
                logger.debug("Using kmeans++ initilization to choose centroids.")
                self.codebook.base[m,:,:] = np.array(kmeans.kmeansplusplus_centers(data_aligned[:, rs:re], use_c=True, use_parallel=parallel))

            elif use_tslearn:
                logger.debug("Performing tsLearn k-means clustering.")
                settings = as_dict(self.kmeans_dist_params)
                del settings["only_ub"]
                kmeans = tslearn_KMeans(
                    metric_params=settings,
                    n_clusters=self.codebook_size,
                    random_state=seed,
                    verbose=0,
                    metric="dtw",
                    max_iter_barycenter=dba_max_iter,
                    max_iter=max_iter,
                    init='k-means++')
                kmeans.fit(to_time_series_dataset(data_aligned[:,rs:re]))
                self.codebook.base[m,:,:] = kmeans.cluster_centers_[:,:,0]

            elif use_kshape or self.metric == Metric.SBD:
                logger.debug("Performing k-shape clustering.")
                kshape = KShape(
                    k=self.codebook_size,
                    max_it=max_iter)
                kshape.fit(np.array(data_aligned[:, rs:re]), use_c=True, use_parallel=parallel)
                self.codebook.base[m,:,:] = np.array(kshape.centroids)

            else:
                logger.debug("Performing dtaidistance k-means clustering.")
                settings = as_dict(self.kmeans_dist_params)
                del settings["only_ub"]
                kmeans = KMeans(
                    k=self.codebook_size,
                    max_it=max_iter,
                    max_dba_it=dba_max_iter,
                    drop_stddev=None,
                    initialize_with_kmedoids=False,
                    initialize_with_kmeanspp=True,
                    dists_options=settings)
                kmeans.fit(np.array(data_aligned[:, rs:re]), use_c=True, use_parallel=parallel)
                # index = nearest_neighbour_lb_keogh(
                #         data_aligned[:,rs:re],
                #         t=np.array(kmeans.means),
                #         use_c=True)
                # self.codebook.base[m,:,:]  = data_aligned[index, rs:re]
                self.codebook.base[m,:,:] = np.array(kmeans.means)
            
            if self.metric == Metric.DTW:
                logger.debug("Precomputing the envelopes of the centroids.")
                for k in range(self.codebook_size):
                    dtaidistancec_dtw_search.lower_upper_ascending_minima(
                            &self.codebook[m,k,0], self.subspace_size + self.tail, 
                            &self.codebookEnvL[m,k,0],
                            &self.codebookEnvU[m,k,0], 
                            &self.quantization_dist_params._settings)        

        # Pre-computations required for the distance strategy
        self._distance_strategy.post_fit(self, parallel)

        # Correct the distances by a factor
        if self.compute_dist_correction:
            self.__determine_dist_correction(data, parallel)

        return self

    def encode(self, data, use_lb_keogh=True, parallel=False):
        """
        Encode a time series.
        """
        cdef double[:,:] s = data
        cdef Py_ssize_t[:,:] codes = self._encode_2D(s, use_lb_keogh, parallel)
        return np.array(codes)

    cdef Py_ssize_t[:] _encode_1D(self, double[:] data, bint use_lb_keogh=True):
        cdef Py_ssize_t[:] codes = np.zeros(self.M, dtype=np.intp)
        if self.subspace_type == SubspaceType.PRE_ALIGN:
            if self.metric == Metric.SBD:
                dtaidistancec_quantizer.encode_sbd(
                        &self.codebook[0,0,0], self.codebook_size, 
                        &data[0], data.shape[0], self.wavelet_level, 
                        self.subspace_size, self.tail, &codes[0])

            else:
                dtaidistancec_quantizer.encode(
                        &self.codebook[0,0,0], self.codebook_size, 
                        &self.codebookEnvL[0,0,0], &self.codebookEnvU[0,0,0], 
                        &data[0], data.shape[0], self.wavelet_level, 
                        self.subspace_size, self.tail, use_lb_keogh, 
                        &self.quantization_dist_params._settings, &codes[0])
        else:
            dtaidistancec_quantizer.encode_overlap(
                &self.codebook[0,0,0], self.codebook_size, 
                &self.codebookEnvL[0,0,0], &self.codebookEnvU[0,0,0], 
                &data[0], data.shape[0],
                self.subspace_size, self.overlap, use_lb_keogh, 
                &self.quantization_dist_params._settings, &codes[0])
        return codes

    cdef Py_ssize_t[:,:] _encode_2D(self, double[:,:] data, bint use_lb_keogh=True, bint parallel=False):
        cdef Py_ssize_t[:,:] codes = np.zeros((data.shape[0], self.M), dtype=np.intp)
        cdef int i
        if parallel:
            if self.subspace_type == SubspaceType.PRE_ALIGN:
                if self.metric == Metric.SBD:
                    for i in prange(data.shape[0], nogil=True):
                        dtaidistancec_quantizer.encode_sbd(
                                &self.codebook[0,0,0], self.codebook_size, 
                                &data[i,0], data.shape[1], self.wavelet_level,
                                self.subspace_size, self.tail, &codes[i,0])
                else:
                    for i in prange(data.shape[0], nogil=True):
                        dtaidistancec_quantizer.encode(
                                &self.codebook[0,0,0], self.codebook_size, 
                                &self.codebookEnvL[0,0,0], &self.codebookEnvU[0,0,0], 
                                &data[i,0], data.shape[1], self.wavelet_level,
                                self.subspace_size, self.tail, use_lb_keogh, 
                                &self.quantization_dist_params._settings, &codes[i,0])
            else:
                for i in prange(data.shape[0], nogil=True):
                    dtaidistancec_quantizer.encode_overlap(
                            &self.codebook[0,0,0], self.codebook_size, 
                            &self.codebookEnvL[0,0,0], &self.codebookEnvU[0,0,0], 
                            &data[i,0], data.shape[1],
                            self.subspace_size, self.overlap, use_lb_keogh, 
                            &self.quantization_dist_params._settings, &codes[i,0])
        else:
            if self.subspace_type == SubspaceType.PRE_ALIGN:
                if self.metric == Metric.SBD:
                    for i in range(data.shape[0]):
                        dtaidistancec_quantizer.encode_sbd(
                            &self.codebook[0,0,0], self.codebook_size, 
                            &data[i,0], data.shape[1], self.wavelet_level,
                            self.subspace_size, self.tail, &codes[i,0])
                else:
                    for i in range(data.shape[0]):
                        dtaidistancec_quantizer.encode(
                            &self.codebook[0,0,0], self.codebook_size, 
                            &self.codebookEnvL[0,0,0], &self.codebookEnvU[0,0,0], 
                            &data[i,0], data.shape[1], self.wavelet_level,
                            self.subspace_size, self.tail, use_lb_keogh, 
                            &self.quantization_dist_params._settings, &codes[i,0])
            else:
                for i in range(data.shape[0]):
                    dtaidistancec_quantizer.encode_overlap(
                            &self.codebook[0,0,0], self.codebook_size, 
                            &self.codebookEnvL[0,0,0], &self.codebookEnvU[0,0,0], 
                            &data[i,0], data.shape[1],
                            self.subspace_size, self.overlap, use_lb_keogh, 
                            &self.quantization_dist_params._settings, &codes[i,0])
        return codes

    cpdef double approx_distance(self, double[::1] s1, double[::1] s2, Py_ssize_t[:] c1=None, Py_ssize_t[:] c2=None):
        """
        Approximate Dynamic Time Warping.

        :param s1: First sequence
        :param s2: Second sequence
        :param c1: Pre-computed encoding of s1
        :param c2: Pre-computed encoding of s2
        :returns: approximate DTW distance
        """
        cdef Py_ssize_t[:] code1 = self._encode_1D(s1) if c1 is None else c1
        cdef Py_ssize_t[:] code2 = self._encode_1D(s2) if c2 is None else c2
        cdef double dist = self._distance_strategy.distance(self, s1, code1, s2, code2)
        return dist

    def approx_distance_matrix_py(self, s, block=None, compact=False, parallel=False, show_progress=False, only_triu=False):
        if isinstance(s, SeriesContainer):
            s = s.series
        return self.approx_distance_matrix(s.copy(order="C"), block, compact, parallel, show_progress, only_triu)

    cpdef approx_distance_matrix(self, double[:,::1] s, block=None, compact=False, parallel=False, show_progress=False, only_triu=False):
        """Approximate distance matrix for all sequences in s.

        :param s: Iterable of series
        :param block: Only compute block in matrix. Expects tuple with begin and end, e.g. ((0,10),(20,25)) will
            only compare rows 0:10 with rows 20:25.
        :param compact: Return the distance matrix as an array representing the upper triangular matrix.
        :param parallel: Use parallel operations
        :param show_progress: Show progress using the tqdm library. This is only supported for
            the non-parallel versions.
        :param only_triu: Only fill the upper triangular part of the distance matrix. Values on the diagonal are infinity.
        :returns: The approximate distance matrix or the condensed distance matrix if the compact argument is true
        """
        cdef Py_ssize_t[:,:] codes = self._encode_2D(s, use_lb_keogh=True, parallel=parallel)
        cdef np.ndarray[np.double_t, ndim = 2] distmat
        cdef int r, c, idx
        cdef array.array dists = array.array('d', [inf] * _distance_matrix_length(block, len(s)))
        cdef double[:] dists_view = dists

        if parallel:
            # TODO
            warnings.warn("Parallel computations are not yet supported.")

        if block is not None:
            if block is None:
                it_r = range(len(s))
            else:
                it_r = range(block[0][0], block[0][1])
            if show_progress:
                it_r = tqdm(it_r)
            idx = 0
            for r in it_r:
                if block is None:
                    it_c = range(r + 1, len(s))
                elif len(block) > 2 and block[2] is False:
                    it_c = range(block[1][0], min(len(s), block[1][1]))
                else:
                    it_c = range(max(r + 1, block[1][0]), min(len(s), block[1][1]))
                for c in it_c:
                    dists_view[idx] = self._distance_strategy.distance(self, s[r], codes[r], s[c], codes[c])
                    idx += 1
        else:
            self._distance_strategy.distance_matrix(self, s, codes, dists_view)

        if compact:
            return dists

        # Create full matrix and fill upper triangular matrix with distance values (or only block if specified)
        distmat = distances_array_to_matrix(dists, nb_series=len(s), block=block, only_triu=only_triu)
        return distmat

    cdef __determine_dist_correction(self, data, parallel=False):
        cdef double sumRatio, cnt = 0.0
        cdef int i, j

        cdef double[:,:] pred = self.approx_distance_matrix(data, parallel=parallel)

        settings = as_dict(self.dist_params)
        del settings["only_ub"]
        cdef double[:,:] truth
        if self.metric == Metric.DTW:
            truth = distance_matrix(data, use_c=True, parallel=parallel, **settings)
        elif self.metric == Metric.SBD:
            truth = sbd_distance_matrix(data, use_c=True, parallel=parallel)

        for i in range(len(pred)):
            for j in range(i+1, len(pred)):
                if pred[i,j] > 0.01 and truth[i,j] > 0.01:
                    sumRatio += truth[i,j] / pred[i,j]
                    cnt += 1
        self.distance_ratio = sumRatio / cnt
        logger.debug ('Distance correction ratio', self.distance_ratio)

    def plot_alignment(self, data, ax=None, **plt_kwargs):

        if not self.M:
            # Set M if this PQ is not yet fitted
            N, D = data.shape
            if self.subspace_type == SubspaceType.NO_OVERLAP:
                self.M = D / self.subspace_size
            elif self.subspace_type == SubspaceType.DOUBLE_OVERLAP:
                self.M = D / self.subspace_size * 2 - 1
            elif self.subspace_type == SubspaceType.PRE_ALIGN:
                self.M = D / self.subspace_size
            else:
                raise ValueError('{} is not a valid type.'.format(self.subspace_type))

        data_aligned = self._prepare_data_2D(data, False)
        if ax is None:
            _, ax = plt.subplots()

        for i in range(data.shape[0]):
            ax.plot(data_aligned[i, :], **plt_kwargs)
        for m in range(self.M + 1):
            ax.axvline(m * (self.subspace_size + self.tail), linestyle='--', color='k', lw=0.5)
        return ax



cdef class PQDistanceStrategy():
    """Implements a strategy to compute the approximate DTW distance."""

    cdef post_fit(self, PQ pq, bint parallel):
        """Executed after fitting the PQ."""
        pass

    cdef double distance(self, PQ pq, double[::1] s1, Py_ssize_t[:] c1, double[::1] s2, Py_ssize_t[:] c2):
        pass

    cdef void distance_matrix(self, PQ pq, double[:,::1] data, Py_ssize_t[:,:] codes, double[:] dists):
        """Computes the approximate distance."""
        pass


cdef class SymmetricPQDistanceStrategy(PQDistanceStrategy):
    """Symetric distance estimation.

    An encoded version of the query is used and precalculated distances can be used
    """
    cdef double[:,:,:] distanceDTWMatrix # The distance between each pair of centroids (M x codebook_size x codebook_size). Used for symmetric distance estimations only.

    cdef post_fit(self, PQ pq, bint parallel):
        self.distanceDTWMatrix = np.empty((pq.M, pq.codebook_size, pq.codebook_size), dtype=np.double)
        cdef int m
        settings = as_dict(pq.dist_params)
        del settings["only_ub"]
        if pq.metric == Metric.DTW:
            for m in range(pq.M):
                self.distanceDTWMatrix.base[m,:,:] = distance_matrix(
                        np.asarray(pq.codebook[m,:,:]),
                        use_c=True,
                        parallel=parallel,
                        compact=False,
                        only_triu=False,
                        **settings)
        if pq.metric == Metric.SBD:
            for m in range(pq.M):
                self.distanceDTWMatrix.base[m,:,:] = sbd_distance_matrix(
                        np.asarray(pq.codebook[m,:,:]),
                        use_c=True,
                        parallel=parallel,
                        compact=False,
                        only_triu=False)


    cdef double distance(self, PQ pq, double[::1] s1, Py_ssize_t[:] c1, double[::1] s2, Py_ssize_t[:] c2):
        cdef int m
        cdef double dist = 0.0
        for m in range(pq.M):
            dist += self.distanceDTWMatrix[m, c1[m], c2[m]]**2 * pq.overlap_corrector
        return sqrt(dist) * pq.distance_ratio

    cdef void distance_matrix(self, PQ pq, double[:,::1] data, Py_ssize_t[:,:] codes, double[:] dists):
        cdef int s = data.shape[0]
        cdef int i, j, m
        cdef Py_ssize_t idx = 0
        for i in range(s):
            for j in range(i+1, s):
                dists[idx] = 0
                for m in range(pq.M):
                    dists[idx] += self.distanceDTWMatrix[m, codes[i,m], codes[j,m]]**2 * pq.overlap_corrector
                dists[idx] = sqrt(dists[idx]) * pq.distance_ratio
                idx += 1


cdef class AsymmetricPQDistanceStrategy(PQDistanceStrategy):
    """Asymetric distance estimation.

    The distance between the centers and the actual query data point is
    calculated to find the nearest neighbors within the indexed queryable
    data. This limits the number of distance calculations to the number of
    codebook entries, which would have been needed for encoding anyway.
    """
    cdef double distance(self, PQ pq, double[::1] s1, Py_ssize_t[:] c1, double[::1] s2, Py_ssize_t[:] c2):
        cdef int m, rs, re
        cdef double dist = 0.0
        cdef double[::1] s1_aligned = pq._prepare_data_1D(s1)
        for m in range(pq.M):
            rs = m * (pq.subspace_size + pq.tail - pq.overlap)
            re = pq.subspace_size + pq.tail

            if pq.metric == Metric.DTW:
                dist += dtaidistancec_dtw.dtw_distance(&s1_aligned[rs], re, &pq.codebook[m,c2[m],0], re, &pq.dist_params._settings)**2 * pq.overlap_corrector
            if pq.metric == Metric.SBD:
                dist += dtaidistancec_sbd.sbd_distance(&s1_aligned[rs], re, &pq.codebook[m,c2[m],0], re)**2 * pq.overlap_corrector
        return sqrt(dist) * pq.distance_ratio

    cdef void distance_matrix(self, PQ pq, double[:,::1] data, Py_ssize_t[:,:] codes, double[:] dists):
        cdef int s = data.shape[0]
        cdef double[:,::1] data_aligned = pq._prepare_data_2D(data, False)
        cdef double[:,:,:] assymDists = self.retrieveAsymDists(pq, data_aligned)
        cdef int xi, ci
        cdef Py_ssize_t idx = 0
        for xi in range(s):
            for ci in range(xi+1, s):
                dists[idx] = 0
                for m in range(pq.M):
                    # approximateMatrix[xi, ci] += dtw_cc.distance(data[xi, rs:re], pq.codebook[m,codes[ci,m],:], **pq.dist_params) **2 * pq.overlap_corrector
                    dists[idx] += assymDists[m, xi, codes[ci, m]]**2 * pq.overlap_corrector
                dists[idx] = sqrt(dists[idx]) * pq.distance_ratio
                idx += 1

    cdef double[:,:,:] retrieveAsymDists(self, PQ pq, double[:,:] data):
        cdef double[:,:,:] distanceDTWMatrix = np.empty((pq.M, data.shape[0], pq.codebook_size), dtype=np.double)
        cdef int m
        settings = as_dict(pq.dist_params)
        del settings["only_ub"]
        for m in range(pq.M):
            rs = m * (pq.subspace_size + pq.tail - pq.overlap)
            re = rs + pq.subspace_size + pq.tail

            if pq.metric == Metric.DTW:
                distanceDTWMatrix.base[m,:,:] = distance_matrix(
                        np.concatenate((data[:,rs:re],  pq.codebook[m,:,:])),
                        block=((0,data.shape[0]), (data.shape[0], data.shape[0]+pq.codebook_size)),
                        use_c=True, **settings
                    )[0:data.shape[0], data.shape[0]:(data.shape[0]+pq.codebook_size)]
            if pq.metric == Metric.SBD:
                distanceDTWMatrix.base[m,:,:] = sbd_distance_matrix(
                        np.concatenate((data[:,rs:re],  pq.codebook[m,:,:])),
                        block=((0,data.shape[0]), (data.shape[0], data.shape[0]+pq.codebook_size)),
                        use_c=True
                    )[0:data.shape[0], data.shape[0]:(data.shape[0]+pq.codebook_size)]

        return distanceDTWMatrix


cdef class ExactWhen0PQDistanceStrategy(SymmetricPQDistanceStrategy):

    cdef double distance(self, PQ pq, double[::1] s1, Py_ssize_t[:] c1, double[::1] s2, Py_ssize_t[:] c2):
        cdef int m, rs, re
        cdef double dist = 0.0
        cdef double[:] s1_aligned
        cdef double[:] s2_aligned
        for m in range(pq.M):
            if c1[m] == c2[m]:
                rs = m * (pq.subspace_size + pq.tail - pq.overlap)
                re = pq.subspace_size + pq.tail
                s1_aligned = pq._prepare_data_1D(s1)
                s2_aligned = pq._prepare_data_1D(s2)

                if pq.metric == Metric.DTW:
                    dist += dtaidistancec_dtw.dtw_distance(&s1_aligned[rs], re, &s2_aligned[rs], re, &pq.dist_params._settings)**2 * pq.overlap_corrector
                if pq.metric == Metric.SBD:
                    dist += dtaidistancec_sbd.sbd_distance(&s1_aligned[rs], re, &s2_aligned[rs], re)**2 * pq.overlap_corrector
            else:
                dist += self.distanceDTWMatrix[m, c1[m], c2[m]]**2 * pq.overlap_corrector
        return sqrt(dist) * pq.distance_ratio

    cdef void distance_matrix(self, PQ pq, double[:,::1] data, Py_ssize_t[:,:] codes, double[:] dists):
        cdef int s = data.shape[0]
        cdef double[:,::1] data_aligned = pq._prepare_data_2D(data, False)
        cdef double[:] si_aligned
        cdef double[:] sj_aligned
        cdef int rs, re, i, j, m
        cdef Py_ssize_t idx = 0
        for i in range(s):
            for j in range(i+1, s):
                dists[idx] = 0
                for m in range(pq.M):
                    if codes[i, m] == codes[j, m]:
                        rs = m * (pq.subspace_size + pq.tail - pq.overlap)
                        re = rs + pq.subspace_size + pq.tail
                        si_aligned = pq._prepare_data_1D(data_aligned[i,:])
                        sj_aligned = pq._prepare_data_1D(data_aligned[j,:])

                        if pq.metric == Metric.DTW:
                            dists[idx] += dtaidistancec_dtw.dtw_distance(&si_aligned[rs], re, &sj_aligned[rs], re, &pq.dist_params._settings)**2 * pq.overlap_corrector
                        if pq.metric == Metric.SBD:
                            dists[idx] += dtaidistancec_sbd.sbd_distance(&si_aligned[rs], re, &sj_aligned[rs], re)**2 * pq.overlap_corrector
                    else:
                        dists[idx] += self.distanceDTWMatrix[m, codes[i,m], codes[j,m]]**2 * pq.overlap_corrector
                dists[idx] = sqrt(dists[idx]) * pq.distance_ratio
                idx += 1


cdef class AsymmetricKeoghWhen0PQDistanceStrategy(SymmetricPQDistanceStrategy):

    cdef double distance(self, PQ pq, double[::1] s1, Py_ssize_t[:] c1, double[::1] s2, Py_ssize_t[:] c2):
        cdef int m, rs, re
        cdef double dist = 0.0
        cdef double[:] s1_aligned
        cdef double[:] s2_aligned
        cdef double lb1, lb2
        for m in range(pq.M):
            if c1[m] == c2[m]:
                rs = m * (pq.subspace_size + pq.tail - pq.overlap)
                re = pq.subspace_size + pq.tail
                s1_aligned = pq._prepare_data_1D(s1)
                s2_aligned = pq._prepare_data_1D(s2)
                lb1 = dtaidistancec_dtw_search.lb_keogh_from_envelope(&s1_aligned[rs], re, &pq.codebookEnvL[m,c2[m],0], &pq.codebookEnvU[m,c2[m],0], &pq.dist_params._settings)
                lb2 = dtaidistancec_dtw_search.lb_keogh_from_envelope(&s2_aligned[rs], re, &pq.codebookEnvL[m,c1[m],0], &pq.codebookEnvU[m,c1[m],0], &pq.dist_params._settings)
                dist += max(lb1, lb2)**2  * pq.overlap_corrector
            else:
                dist += self.distanceDTWMatrix[m, c1[m], c2[m]]**2 * pq.overlap_corrector
        return sqrt(dist) * pq.distance_ratio

    cdef void distance_matrix(self, PQ pq, double[:,::1] data, Py_ssize_t[:,:] codes, double[:] dists):
        cdef int s = data.shape[0]
        cdef double[:,::1] data_aligned = pq._prepare_data_2D(data, False)
        cdef double[:] si_aligned
        cdef double[:] sj_aligned
        cdef int rs, re, i, j, m
        cdef double lbi, lbj
        cdef Py_ssize_t idx = 0
        for i in range(s):
            for j in range(i+1, s):
                dists[idx] = 0
                for m in range(pq.M):
                    if codes[i,m] == codes[j,m]:
                        rs = m * (pq.subspace_size + pq.tail - pq.overlap)
                        re = pq.subspace_size + pq.tail
                        si_aligned = pq._prepare_data_1D(data_aligned[i,:])
                        sj_aligned = pq._prepare_data_1D(data_aligned[j,:])
                        lbi = dtaidistancec_dtw_search.lb_keogh_from_envelope(&si_aligned[rs], re, &pq.codebookEnvL[m,codes[j,m],0], &pq.codebookEnvU[m,codes[j,m],0], &pq.dist_params._settings)
                        lbj = dtaidistancec_dtw_search.lb_keogh_from_envelope(&sj_aligned[rs], re, &pq.codebookEnvL[m,codes[i,m],0], &pq.codebookEnvU[m,codes[i,m],0], &pq.dist_params._settings)
                        dists[idx] += max(lbi, lbj)**2  * pq.overlap_corrector
                    else:
                        dists[idx] += self.distanceDTWMatrix[m, codes[i,m], codes[j,m]]**2  * pq.overlap_corrector
                dists[idx] = sqrt(dists[idx]) * pq.distance_ratio
                idx += 1
