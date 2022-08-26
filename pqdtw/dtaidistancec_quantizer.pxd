
from dtaidistancec_globals cimport seq_t
from dtaidistancec_dtw cimport DTWSettings


cdef extern from "dd_quantizer.h":
    int get_wavelet_splits(seq_t *inp, Py_ssize_t l, int J, Py_ssize_t subspace_size, Py_ssize_t tail, seq_t *out) nogil
    int encode_overlap(seq_t *codebook, Py_ssize_t codebook_size, seq_t *lenv, seq_t *uenv, seq_t *s, Py_ssize_t l, Py_ssize_t subspace_size, int overlap, int use_lb_keogh, DTWSettings *settings, Py_ssize_t *codes) nogil
    int encode(seq_t *codebook, Py_ssize_t codebook_size, seq_t *lenv, seq_t *uenv, seq_t *s, Py_ssize_t l, int J, Py_ssize_t subspace_size, Py_ssize_t tail, int use_lb_keogh, DTWSettings *settings, Py_ssize_t *codes) nogil
    int encode_sbd(seq_t *codebook, Py_ssize_t codebook_size, seq_t *s, Py_ssize_t l, int J, Py_ssize_t subspace_size, Py_ssize_t tail, Py_ssize_t *codes) nogil
