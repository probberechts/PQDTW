#ifndef quantizer_h
#define quantizer_h

#include "dd_globals.h"
#include "dd_dtw_search.h"

int get_wavelet_splits(seq_t *inp, idx_t l, int J, idx_t subspace_size, idx_t tail, seq_t *out);
int encode_overlap(seq_t *codebook, idx_t codebook_size, seq_t *lenv, seq_t *uenv, seq_t *s, idx_t l, idx_t subspace_size, int overlap, int use_lb_keogh, DTWSettings *settings, idx_t *codes);
int encode(seq_t *codebook, idx_t codebook_size, seq_t *lenv, seq_t *uenv, seq_t *s, idx_t l, int J, idx_t subspace_size, idx_t tail, int use_lb_keogh, DTWSettings *settings, idx_t *codes);
int encode_sbd(seq_t *codebook, idx_t codebook_size, seq_t *s, idx_t l, int J, idx_t subspace_size, idx_t tail, idx_t *codes);

#endif /* quantizer_h */
