"""
dtaidistance.sbd_cc
~~~~~~~~~~~~~~~~~~

Shape-Based Distance (SBD), C implementation.

:author: Pieter Robberechts
:copyright: Copyright 2021 KU Leuven, DTAI Research Group.
:license: Apache License, Version 2.0, see LICENSE for details.

"""
import logging
from cpython cimport array
import array
from dtw_cc cimport DTWSeriesMatrix, DTWSeriesMatrixNDim, DTWSeriesPointers, DTWBlock
from dtw_cc import dtw_series_from_data, distance_matrix_length

cimport dtaidistancec_sbd
cimport dtaidistancec_dtw_omp


logger = logging.getLogger("be.kuleuven.dtai.pqdtw")

def is_openmp_supported():
    return dtaidistancec_dtw_omp.is_openmp_supported()


def NCCc(double[:] s1, double[:] s2):
    """ Cross-correlation with coefficient normalization

    :param s1: Sequence of numbers
    :param s2: Sequence of numbers
    :return: The cross-correlation sequence with length `len(s1) + len(s2) - 1`.
    """
    cdef array.array out = array.array('d')
    array.resize(out, len(s1) + len(s2) + 1)
    dtaidistancec_sbd.NCCc(&s1[0], len(s1), &s2[0], len(s2), out.data.as_doubles)
    return out


def distance(double[:] s1, double[:] s2):
    """ Shape-based distance between two sequences. Supports different lengths.

    :param s1: Sequence of numbers
    :param s2: Sequence of numbers
    :return: Shape-based distance
    """
    return dtaidistancec_sbd.sbd_distance(&s1[0], len(s1), &s2[0], len(s2))


def distance_matrix(cur, block=None):
    """Compute a distance matrix between all sequences given in `cur`.
    This method calls a pure c implementation of the dtw computation that
    avoids the GIL.

    Assumes C-contiguous arrays.

    :param cur: DTWSeriesMatrix or DTWSeriesPointers
    :param block: see DTWBlock
    :return: The distance matrix as a list representing the triangular matrix.
    """
    cdef DTWSeriesMatrix matrix
    cdef DTWSeriesPointers ptrs
    cdef Py_ssize_t length = 0
    cdef Py_ssize_t block_rb=0
    cdef Py_ssize_t block_re=0
    cdef Py_ssize_t block_cb=0
    cdef Py_ssize_t block_ce=0
    cdef Py_ssize_t ri = 0
    if block is not None and block != 0.0:
        block_rb = block[0][0]
        block_re = block[0][1]
        block_cb = block[1][0]
        block_ce = block[1][1]

    cdef DTWBlock dtwblock = DTWBlock(rb=block_rb, re=block_re, cb=block_cb, ce=block_ce)
    if block is not None and block != 0.0 and len(block) > 2 and block[2] is False:
        dtwblock.triu_set(False)
    length = distance_matrix_length(dtwblock, len(cur))

    # Correct block
    if dtwblock.re == 0:
        dtwblock.re_set(len(cur))
    if dtwblock.ce == 0:
        dtwblock.ce_set(len(cur))

    cdef array.array dists = array.array('d')
    array.resize(dists, length)

    if isinstance(cur, DTWSeriesMatrix) or isinstance(cur, DTWSeriesPointers):
        pass
    elif cur.__class__.__name__ == "SeriesContainer":
        cur = cur.c_data_compat()
    else:
        cur = dtw_series_from_data(cur)

    if isinstance(cur, DTWSeriesPointers):
        ptrs = cur
        dtaidistancec_sbd.sbd_distances_ptrs(
            ptrs._ptrs, ptrs._nb_ptrs, ptrs._lengths,
            dists.data.as_doubles, &dtwblock._block)
    elif isinstance(cur, DTWSeriesMatrix):
        matrix = cur
        dtaidistancec_sbd.sbd_distances_matrix(
            &matrix._data[0,0], matrix.nb_rows, matrix.nb_cols,
            dists.data.as_doubles, &dtwblock._block)

    return dists


def distance_matrix_parallel(cur, block=None):
    """Compute a distance matrix between all sequences given in `cur`.
    This method calls a pure c implementation of the sbd computation that
    avoids the GIL.

    Assumes C-contiguous arrays.

    :param cur: DTWSeriesMatrix or DTWSeriesPointers
    :param block: see DTWBlock
    :return: The distance matrix as a list representing the triangular matrix.
    """
    cdef DTWSeriesMatrix matrix
    cdef DTWSeriesPointers ptrs
    cdef Py_ssize_t length = 0
    cdef Py_ssize_t block_rb=0
    cdef Py_ssize_t block_re=0
    cdef Py_ssize_t block_cb=0
    cdef Py_ssize_t block_ce=0
    cdef Py_ssize_t ri = 0
    if block is not None and block != 0.0:
        block_rb = block[0][0]
        block_re = block[0][1]
        block_cb = block[1][0]
        block_ce = block[1][1]

    cdef DTWBlock dtwblock = DTWBlock(rb=block_rb, re=block_re, cb=block_cb, ce=block_ce)
    if block is not None and block != 0.0 and len(block) > 2 and block[2] is False:
        dtwblock.triu_set(False)
    length = distance_matrix_length(dtwblock, len(cur))

    # Correct block
    if dtwblock.re == 0:
        dtwblock.re_set(len(cur))
    if dtwblock.ce == 0:
        dtwblock.ce_set(len(cur))

    cdef array.array dists = array.array('d')
    array.resize(dists, length)

    if isinstance(cur, DTWSeriesMatrix) or isinstance(cur, DTWSeriesPointers):
        pass
    elif cur.__class__.__name__ == "SeriesContainer":
        cur = cur.c_data_compat()
    else:
        cur = dtw_series_from_data(cur)

    if isinstance(cur, DTWSeriesPointers):
        ptrs = cur
        dtaidistancec_sbd.sbd_distances_ptrs_parallel(
            ptrs._ptrs, ptrs._nb_ptrs, ptrs._lengths,
            dists.data.as_doubles, &dtwblock._block)
    elif isinstance(cur, DTWSeriesMatrix):
        matrix = cur
        dtaidistancec_sbd.sbd_distances_matrix_parallel(
            &matrix._data[0,0], matrix.nb_rows, matrix.nb_cols,
            dists.data.as_doubles, &dtwblock._block)

    return dists


