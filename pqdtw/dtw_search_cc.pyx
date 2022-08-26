#cython: wraparound=False
#cython: boundscheck=False
#cython: nonecheck=False

"""
dtaidistance.dt_search_cc
~~~~~~~~~~~~~~~~~~~~~~~~~

DTW-based nearest neighbor search, C implementation.

:author: Pieter Robberechts
:copyright: Copyright 2021 KU Leuven, DTAI Research Group.
:license: Apache License, Version 2.0, see LICENSE for details.

"""
from cpython cimport array
import array
from cython import Py_ssize_t
from cython.view cimport array as cvarray
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from cython.parallel import parallel, prange

from dtaidistance.dtw_cc cimport DTWSeriesMatrix, DTWSettings

cimport dtaidistancec_dtw_search
cimport dtaidistancec_dtw


cdef extern from "Python.h":
    Py_ssize_t PY_SSIZE_T_MAX


def lb_keogh_envelope(double[:] s1, **kwargs):
    cdef Py_ssize_t length = s1.shape[0]
    settings = DTWSettings(**kwargs)

    cdef array.array L = array.array('d')
    array.resize(L, length)
    cdef array.array U = array.array('d')
    array.resize(U, length)

    dtaidistancec_dtw_search.lower_upper_ascending_minima(&s1[0], len(s1), L.data.as_doubles, U.data.as_doubles, &settings._settings)
    return L, U


def lb_keogh_envelope_parallel(s1, **kwargs):
    settings = DTWSettings(**kwargs)

    cdef DTWSeriesMatrix matrix = s1.c_data_compat()
    L = cvarray(shape=(matrix._data.shape[0], matrix._data.shape[1]), itemsize=sizeof(double), format="d")
    cdef double [:, :] L_view = L
    U = cvarray(shape=(matrix._data.shape[0], matrix._data.shape[1]), itemsize=sizeof(double), format="d")
    cdef double [:, :] U_view = U

    cdef int d
    for d in prange(matrix._data.shape[0], nogil=True):
        dtaidistancec_dtw_search.lower_upper_ascending_minima(&matrix._data[d,0], matrix._data.shape[1], &L_view[d,0], &U_view[d,0], &settings._settings)
    return list(zip(list(L), list(U)))


def lb_keogh_from_envelope(double[:] s1, s2_envelope, **kwargs):
    settings = DTWSettings(**kwargs)
    cdef double[:] L = s2_envelope[0]
    cdef double[:] U = s2_envelope[1]
    cdef double lb = dtaidistancec_dtw_search.lb_keogh_from_envelope(&s1[0], len(s1), &L[0], &U[0], &settings._settings)
    return lb


def lb_keogh_from_envelope_parallel(s1,  s2_envelope, **kwargs):
    settings = DTWSettings(**kwargs)

    cdef DTWSeriesMatrix matrix1 = s1.c_data_compat()
    cdef int l = matrix1._data.shape[1]
    cdef int l1 = matrix1._data.shape[0]
    cdef int l2 = len(s2_envelope)
    cdef int i, j
    lb = cvarray(shape=(l1, l2), itemsize=sizeof(double), format="d")
    cdef double[:,:] lb_view = lb
    L = cvarray(shape=(l2, l), itemsize=sizeof(double), format="d")
    cdef double [:, :] L_view = L
    U = cvarray(shape=(l2, l), itemsize=sizeof(double), format="d")
    cdef double [:, :] U_view = U
    for i in range(l2):
        for j in range(l):
            L_view[i, j] = s2_envelope[i][0][j]
            U_view[i, j] = s2_envelope[i][1][j]

    with nogil:
        for i in prange(l1):
            for j in range(l2):
                lb_view[i,j] = dtaidistancec_dtw_search.lb_keogh_from_envelope(
                        &matrix1._data[i,0], matrix1._data.shape[1], 
                        &L_view[j,0], &U_view[j,0], &settings._settings)
    return lb


def lb_keogh(double[:] s1,  double[:] s2, **kwargs):
    settings = DTWSettings(**kwargs)
    cdef double lb = dtaidistancec_dtw.lb_keogh(&s1[0], len(s1), &s2[0], len(s2), &settings._settings)
    return lb


def lb_keogh_parallel(s1, s2, **kwargs):
    settings = DTWSettings(**kwargs)

    cdef DTWSeriesMatrix matrix1 = s1.c_data_compat()
    cdef int l1 = matrix1._data.shape[0]
    cdef DTWSeriesMatrix matrix2 = s2.c_data_compat()
    cdef int l2 = matrix2._data.shape[0]

    lb = cvarray(shape=(l1, l2), itemsize=sizeof(double), format="d")
    cdef double[:,:] lb_view = lb

    cdef int i, j
    for i in prange(l1, nogil=True):
        for j in range(l2):
            lb_view[i,j] = dtaidistancec_dtw.lb_keogh(
                    &matrix1._data[i,0], matrix1._data.shape[1], 
                    &matrix2._data[j,0], matrix2._data.shape[1], 
                    &settings._settings)
    return lb


def nearest_neighbour_lb_keogh(int K, data, double[:] query, envelope, **kwargs):
    cdef int verbose = 0

    cdef Py_ssize_t *locations_view = <Py_ssize_t *> PyMem_Malloc(K * sizeof(Py_ssize_t))
    if not locations_view:
        raise MemoryError()
    cdef array.array locations = array.array('i')
    array.resize(locations, K)
    cdef array.array distances = array.array('d')
    array.resize(distances, K)

    settings = DTWSettings(**kwargs)
    cdef DTWSeriesMatrix matrix = data.c_data_compat()
    cdef double[:] L = envelope[0]
    cdef double[:] U = envelope[1]

    try:
        dtaidistancec_dtw_search.nn_ucrdtw(K,
            &query[0], query.shape[0], 
            &L[0], &U[0],
            &matrix._data[0,0], matrix._data.shape[0], 
            verbose, &locations_view[0], distances.data.as_doubles, &settings._settings)
        if K == 1:
            return locations_view[0], distances[0]
        for i in range(K):
            locations[i] = locations_view[i]
        return locations, distances
    finally:
        PyMem_Free(locations_view)


def nearest_neighbour_lb_keogh_parallel(int K, data, queries, envelopes, **kwargs):
    cdef int verbose = 0
    cdef Py_ssize_t i

    settings = DTWSettings(**kwargs)

    cdef DTWSeriesMatrix qmatrix = queries.c_data_compat()
    cdef int nq = qmatrix._data.shape[0]
    cdef int lq = qmatrix._data.shape[1]

    L = cvarray(shape=(nq, lq), itemsize=sizeof(double), format="d")
    cdef double [:, :] L_view = L
    U = cvarray(shape=(nq, lq), itemsize=sizeof(double), format="d")
    cdef double [:, :] U_view = U
    for i in range(nq):
        for j in range(lq):
            L_view[i, j] = envelopes[i][0][j]
            U_view[i, j] = envelopes[i][1][j]

    cdef Py_ssize_t *locations_view = <Py_ssize_t *> PyMem_Malloc(nq*K * sizeof(Py_ssize_t))
    if not locations_view:
        raise MemoryError()

    distances = cvarray(shape=(nq, K), itemsize=sizeof(double), format="d")
    cdef double[:,:] distances_view = distances

    cdef DTWSeriesMatrix matrix = data.c_data_compat()

    try:
        for i in prange(nq, nogil=True):
            dtaidistancec_dtw_search.nn_ucrdtw(K,
                &qmatrix._data[i,0], lq, &L_view[i,0], &U_view[i,0],
                &matrix._data[0,0], matrix._data.shape[0], 
                verbose, &locations_view[i*K], &distances_view[i,0],
                &settings._settings)
        if K == 1:
            locations = cvarray(shape=(nq,), itemsize=sizeof(int), format="i")
            for i in range(nq):
                locations[i] = locations_view[i]
        else:
            locations = cvarray(shape=(nq, K), itemsize=sizeof(int), format="i")
            for i in range(nq):
                for k in range(K):
                    locations[i,k] = locations_view[i*K+k]
        return locations, distances
    finally:
        PyMem_Free(locations_view)


# def pqknn(int k, distfn, double[:,:] data, double[:,:] queries, int[:,:] dataenc, int[:,:] queriesenc, double[:,:] lb):
#     cdef int i, j
#     cdef double score
#     cdef int[:,:] results = np.zeros((queries.shape[0],k), dtype=np.int32)
#     # results = []
#     cdef double bsf
#     cdef int[:] best_index
#     cdef double[:] best_dist
#     for i in range(queries.shape[0]):
#         bsf = float('inf')
#         best_index = np.ones(k, dtype=np.int32)* -1
#         best_dist = np.ones(k) * float('inf')
#         for j in range(data.shape[0]):
#             if bsf > lb[i,j]:
#                 score = distfn(queries[i,:], data[j,:], queriesenc[i,:], dataenc[j,:])
#                 if score < bsf:
#                     d = np.argmax(best_dist)
#                     best_dist[d] = score
#                     best_index[d] = j
#                     bsf = score
#         for j in range(k):
#             results[i,:] = np.array(best_index)
#         # results.append([int(bsf[1]) for bsf in best_so_far])
#     return results
