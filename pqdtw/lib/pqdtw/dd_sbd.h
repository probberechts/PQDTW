/*!
@header sbd.h
@brief DTAIDistance.ed : Shape-Based Distance

@author Pieter Robberechts
@copyright Copyright © 2020 Wannes Meert. Apache License, Version 2.0, see LICENSE for details.
*/

#ifndef sbd_h
#define sbd_h

#include <math.h>
#include <stdio.h>
#include <assert.h>
#if defined(_OPENMP)
#include <omp.h>
#endif


#include "dd_globals.h"
#include "dd_dtw.h"
#include "dd_dtw_openmp.h"
#include "../pocketfft/pocketfft.h"

int NCCc(seq_t *s1, idx_t l1, seq_t *s2, idx_t l2, seq_t *out);
seq_t sbd_distance(seq_t *s1, idx_t l1, seq_t *s2, idx_t l2);
seq_t sbd_distance_ndim(seq_t *s1, idx_t l1, seq_t *s2, idx_t l2, int ndim);
idx_t sbd_distances_ptrs(seq_t **ptrs, idx_t nb_ptrs, idx_t* lengths, seq_t* output, DTWBlock* block);
idx_t sbd_distances_matrix(seq_t *matrix, idx_t nb_rows, idx_t nb_cols, seq_t* output, DTWBlock* block);
idx_t sbd_distances_ptrs_parallel(seq_t **ptrs, idx_t nb_ptrs, idx_t* lengths, seq_t* output, DTWBlock* block);
idx_t sbd_distances_matrix_parallel(seq_t *matrix, idx_t nb_rows, idx_t nb_cols, seq_t* output, DTWBlock* block);

#endif /* ed_h */
