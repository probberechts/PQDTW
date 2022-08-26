
from dtaidistancec_globals cimport seq_t
from dtaidistancec_dtw cimport DTWBlock


cdef extern from "dd_sbd.h":
    int NCCc(seq_t *s1, Py_ssize_t l1, seq_t *s2, Py_ssize_t l2, seq_t *out)
    seq_t sbd_distance(seq_t *s1, Py_ssize_t l1, seq_t *s2, Py_ssize_t l2)
    seq_t sbd_distance_ndim(seq_t *s1, Py_ssize_t l1, seq_t *s2, Py_ssize_t l2, int ndim)
    Py_ssize_t sbd_distances_ptrs(seq_t **ptrs, Py_ssize_t nb_ptrs, Py_ssize_t* lengths, seq_t* output, DTWBlock* block)
    Py_ssize_t sbd_distances_matrix(seq_t *matrix, Py_ssize_t nb_rows, Py_ssize_t nb_cols, seq_t* output, DTWBlock* block)
    Py_ssize_t sbd_distances_ptrs_parallel(seq_t **ptrs, Py_ssize_t nb_ptrs, Py_ssize_t* lengths, seq_t* output, DTWBlock* block)
    Py_ssize_t sbd_distances_matrix_parallel(seq_t *matrix, Py_ssize_t nb_rows, Py_ssize_t nb_cols, seq_t* output, DTWBlock* block)
