from dtaidistancec_globals cimport seq_t
from dtaidistancec_dtw cimport DTWSettings

cdef extern from "dd_dtw_search.h":

    void lower_upper_naive(seq_t *t, Py_ssize_t len, seq_t *l, seq_t *u, DTWSettings *settings) nogil
    void lower_upper_ascending_minima(seq_t *t, Py_ssize_t len, seq_t *l, seq_t *u, DTWSettings *settings) nogil
    seq_t lb_keogh_from_envelope(seq_t *s1, Py_ssize_t l1, seq_t *l, seq_t *u, DTWSettings *settings) nogil
    int nn_naive(int K, seq_t *query, Py_ssize_t query_size, seq_t *data, Py_ssize_t data_size, int verbose, Py_ssize_t *location, seq_t *distance, DTWSettings *settings) nogil
    int nn_ucrdtw(int K, seq_t *query, Py_ssize_t query_size, seq_t *l, seq_t *u, seq_t *data, Py_ssize_t data_size, int verbose, Py_ssize_t *location, seq_t *distance, DTWSettings *settings) nogil
