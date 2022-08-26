# -*- coding: UTF-8 -*-
"""
dtaidistance.sbd
~~~~~~~~~~~~~~~~

Shape-Based Distance (SBD)

:author: Pieter Robberechts
:copyright: Copyright 2021 KU Leuven, DTAI Research Group.
:license: Apache License, Version 2.0, see LICENSE for details.

"""
import logging
import array

from dtaidistance import util_numpy
from dtaidistance.util import SeriesContainer
from dtaidistance.dtw import _distance_matrix_length, _distance_matrix_idxs, distances_array_to_matrix


logger = logging.getLogger("be.kuleuven.dtai.distance")

sbd_cc = None
try:
    from . import sbd_cc
except ImportError:
    logger.debug('DTAIDistance C library not available')
    sbd_cc = None
try:
    if util_numpy.test_without_numpy():
        raise ImportError()
    import numpy as np
except ImportError:
    np = None


try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

inf = float("inf")

def _check_library(include_omp=False, raise_exception=True):
    if sbd_cc is None:
        msg = "The compiled dtaidistance C library is not available.\n" + \
              "See the documentation for alternative installation options."
        logger.error(msg)
        if raise_exception:
            raise Exception(msg)
    if include_omp and (sbd_cc is None or not sbd_cc.is_openmp_supported()):
        msg = "The compiled dtaidistance C-OMP library "
        if sbd_cc and not sbd_cc.is_openmp_supported():
            msg += "indicates that OpenMP was not avaiable during compilation.\n"
        else:
            msg += "is not available.\n"
        msg += "Use Python's multiprocessing library for parellelization (use_mp=True).\n" + \
               "Call dtw.try_import_c() to get more verbose errors.\n" + \
               "See the documentation for alternative installation options."
        logger.error(msg)
        if raise_exception:
            raise Exception(msg)

def roll_zeropad(a, shift, axis=None):
    a = np.asanyarray(a)
    if shift == 0:
        return a
    if axis is None:
        n = a.size
        reshape = True
    else:
        n = a.shape[axis]
        reshape = False
    if np.abs(shift) > n:
        res = np.zeros_like(a)
    elif shift < 0:
        shift += n
        zeros = np.zeros_like(a.take(np.arange(n-shift), axis))
        res = np.concatenate((a.take(np.arange(n-shift, n), axis), zeros), axis)
    else:
        zeros = np.zeros_like(a.take(np.arange(n-shift, n), axis))
        res = np.concatenate((zeros, a.take(np.arange(n-shift), axis)), axis)
    if reshape:
        return res.reshape(a.shape)
    else:
        return res


def NCCc(s1, s2):
    """
    Cross-correlation with coefficient normalization

    This function uses the FFT to compute the cross-correlation sequence
    between two series. They need *not* be of equal length.

    :param s1: First sequence
    :param s2: Second sequence

    :returns: The cross-correlation sequence with length `len(s1) + len(s2) - 1`.
    """
    den = np.array(np.linalg.norm(s1) * np.linalg.norm(s2))
    den[den == 0] = np.Inf

    s1_len = len(s1)
    fft_size = 1 << (2*s1_len-1).bit_length()
    cc = np.fft.ifft(np.fft.fft(s1, fft_size) * np.conj(np.fft.fft(s2, fft_size)))
    cc = np.concatenate((cc[-(s1_len-1):], cc[:s1_len]))
    return np.real(cc) / den


def distance(s1, s2, use_c=False):
    """
    Shape-Based Distance.

    Distance based on coefficient-normalized cross-correlation as proposed by
    Paparrizos and Gravano (2015) for the k-Shape clustering algorithm.

    :param s1: First sequence
    :param s2: Second sequence
    :param use_c: Use fast pure c compiled functions

    :returns: SBD distance
    """
    if use_c:
        if sbd_cc is None:
            logger.warning("C-library not available, using the Python version")
        else:
            return distance_fast(s1, s2)

    ncc = NCCc(s1, s2)
    return 1 - np.amax(ncc)

def align(s1, s2, use_c=False):
    """
    Align two sequences.

    :param s1: First sequence
    :param s2: Second sequence
    :param use_c: Use fast pure c compiled functions
    :returns: Aligned sequence s2' of s2 towards s1
    """
    if use_c:
        _check_library(raise_exception=True)
        # Check that Numpy arrays for C contiguous
        s1 = util_numpy.verify_np_array(s1)
        s2 = util_numpy.verify_np_array(s2)
        ncc = np.array(sbd_cc.NCCc(s1, s2))
    else:
        ncc = NCCc(s1, s2)
    idx = ncc.argmax()
    yshifted = roll_zeropad(s2, (idx + 1) - max(len(s1), len(s2)))
    return yshifted


def distance_fast(s1, s2):
    """
    Same as :meth:`distance` but with different defaults to chose the fast
    C-based version of the implementation (use_c = True).

    Note: the series are expected to be arrays of the type ``double``.
    Thus ``numpy.array([1,2,3], dtype=numpy.double)`` or 
    ``array.array('d', [1,2,3])``
    """
    _check_library(raise_exception=True)
    # Check that Numpy arrays for C contiguous
    s1 = util_numpy.verify_np_array(s1)
    s2 = util_numpy.verify_np_array(s2)
    # Move data to C library
    return sbd_cc.distance(s1, s2)


def distance_matrix(s, block=None, compact=False, parallel=False, use_c=False,
    use_mp=False, show_progress=False, only_triu=False):
    """
    Distance matrix for all sequences in s.

    :param s: Iterable of series
    :param block: Only compute block in matrix. Expects tuple with begin and end, e.g. ((0,10),(20,25)) will
        only compare rows 0:10 with rows 20:25.
    :param compact: Return the distance matrix as an array representing the upper triangular matrix.
    :param parallel: Use parallel operations
    :param use_c: Use c compiled Python functions
    :param use_mp: Force use Multiprocessing for parallel operations (not OpenMP)
    :param show_progress: Show progress using the tqdm library. This is only supported for
        the pure Python version (thus not the C-based implementations).
    :returns: The distance matrix or the condensed distance matrix if the compact argument is true
    """
    # Check whether multiprocessing is available
    if use_c:
        requires_omp = parallel and not use_mp
        _check_library(raise_exception=True, include_omp=requires_omp)
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
    if block is not None:
        if len(block) > 2 and block[2] is False and compact is False:
            raise Exception(f'Block cannot have a third argument triu=false with compact=false')
        if (block[0][1] - block[0][0]) < 1 or (block[1][1] - block[1][0]) < 1:
            return []
    # Prepare options and data to pass to distance method
    dist_opts = {}
    s = SeriesContainer.wrap(s)
    dists = None
    if use_c:
        for k, v in dist_opts.items():
            if v is None:
                # None is represented as 0.0 for C
                dist_opts[k] = 0

    logger.info('Computing distances')
    if use_c and parallel and not use_mp and sbd_cc is not None:
        logger.info("Compute distances in C (parallel=OMP)")
        dist_opts['block'] = block
        dists = sbd_cc.distance_matrix(s, **dist_opts)

    elif use_c and parallel and (sbd_cc is None or use_mp):
        logger.info("Compute distances in C (parallel=MP)")
        idxs = _distance_matrix_idxs(block, len(s))
        with mp.Pool() as p:
            dists = p.map(_distance_c_with_params, [(s[r], s[c], dist_opts) for c, r in zip(*idxs)])

    elif use_c and not parallel:
        logger.info("Compute distances in C (parallel=No)")
        dist_opts['block'] = block
        dists = sbd_cc.distance_matrix(s, **dist_opts)

    elif not use_c and parallel:
        logger.info("Compute distances in Python (parallel=MP)")
        idxs = _distance_matrix_idxs(block, len(s))
        with mp.Pool() as p:
            dists = p.map(_distance_with_params, [(s[r], s[c], dist_opts) for c, r in zip(*idxs)])

    elif not use_c and not parallel:
        logger.info("Compute distances in Python (parallel=No)")
        dists = distance_matrix_python(s, block=block, show_progress=show_progress)

    else:
        raise Exception(f'Unsupported combination of: parallel={parallel}, '
                        f'use_c={use_c}, sbd_cc={sbd_cc}, use_mp={use_mp}')

    exp_length = _distance_matrix_length(block, len(s))
    assert len(dists) == exp_length, "len(dists)={} != {}".format(len(dists), exp_length)
    if compact:
        return dists

    # Create full matrix and fill upper triangular matrix with distance values (or only block if specified)
    dists_matrix = distances_array_to_matrix(dists, nb_series=len(s), block=block, only_triu=only_triu)

    return dists_matrix


def distance_matrix_python(s, block=None, show_progress=False):
    dists = array.array('d', [inf] * _distance_matrix_length(block, len(s)))
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
            dists[idx] = distance(s[r], s[c])
            idx += 1
    return dists


def _distance_with_params(t):
    return distance(t[0], t[1], **t[2])


def _distance_c_with_params(t):
    return sbd_cc.distance(t[0], t[1], **t[2])
