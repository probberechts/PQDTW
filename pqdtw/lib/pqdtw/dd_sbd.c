//
//  sbd.c
//  DTAIDistanceC
//
//  Created by Pieter Robberechts on 05/10/2021.
//  Copyright © 2020 Wannes Meert. All rights reserved.
//

#include <string.h>
#include "dd_sbd.h"


seq_t norm(seq_t *s, idx_t l) {
    seq_t accum = 0.;
    for (idx_t i = 0; i < l; i++) {
        accum += s[i] * s[i];
    }
    return sqrt(accum);
}

void ascomplex(seq_t *in, idx_t l, seq_t *out) {
    for (idx_t i = 0; i < l; i++ ) {
      out[i*2] = in[i];
    }
}


int NCCc(seq_t *s1, idx_t l1, seq_t *s2, idx_t l2, seq_t *out) {
    seq_t normed =  norm(s1, l1) * norm(s2, l2);
    if (normed == 0) {
        normed =  1000000000000.0;
    }

    int fft_size = 1 << (int)ceil(log2((2*l1 - 1)));

    cfft_plan plan = make_cfft_plan(fft_size);
    seq_t* s1_fft = (seq_t*)calloc(2*fft_size, sizeof(seq_t));
    ascomplex(s1, l1, s1_fft);
    cfft_forward(plan, s1_fft, 1.);

    seq_t* s2_fft = (seq_t*)calloc(2*fft_size, sizeof(seq_t));
    ascomplex(s2, l2, s2_fft);
    cfft_forward(plan, s2_fft, 1.);

    seq_t* inter = (seq_t*)malloc(2*fft_size * sizeof(seq_t));
    for (idx_t i = 0; i < 2*fft_size; i+=2 ) {
      inter[i] = s1_fft[i] * s2_fft[i] - s1_fft[i+1] * -s2_fft[i+1];
      inter[i+1] = s1_fft[i] * -s2_fft[i+1] + s2_fft[i] * s1_fft[i+1];
    }
    cfft_backward(plan, inter, 1./fft_size);
    destroy_cfft_plan (plan);

    for (idx_t i = 0; i < l1; i++) {
      out[i] = inter[2 * (fft_size - l1 + i)] / normed;
    }
    for (idx_t i = 0; i <= l1; i++) {
      out[i+l1] = inter[2 * i] / normed;
    }

    free(inter);
    free(s1_fft);
    free(s2_fft);

    return 0;
}


seq_t sbd_distance(seq_t *s1, idx_t l1, seq_t *s2, idx_t l2) {
    idx_t r_len = l1 + l2 + 1;
    seq_t* ncc = malloc(r_len * sizeof(seq_t));
    NCCc(s1, l1, s2, l2, ncc);

    seq_t max = ncc[0];
    for (int i = 0; i < r_len; i++) {
        if (max < ncc[i])
            max = ncc[i];
    }
    free(ncc);

    return 1.0 - max;
}


seq_t sbd_distance_ndim(seq_t *s1, idx_t l1, seq_t *s2, idx_t l2, int ndim) {
  return 0;
}

idx_t sbd_distances_ptrs(seq_t **ptrs, idx_t nb_ptrs, idx_t* lengths, seq_t* output, DTWBlock* block) {
    idx_t r, c, cb;
    idx_t length;
    idx_t i;
    seq_t value;
    
    length = dtw_distances_length(block, nb_ptrs);
    if (length == 0) {
        return 0;
    }
    
    // Correct block
    if (block->re == 0) {
        block->re = nb_ptrs;
    }
    if (block->ce == 0) {
        block->ce = nb_ptrs;
    }

    i = 0;
    for (r=block->rb; r<block->re; r++) {
        if (block->triu && r + 1 > block->cb) {
            cb = r+1;
        } else {
            cb = block->cb;
        }
        for (c=cb; c<block->ce; c++) {
            value = sbd_distance(ptrs[r], lengths[r], ptrs[c], lengths[c]);
//            printf("i=%zu - r=%zu - c=%zu - value=%.4f\n", i, r, c, value);
            output[i] = value;
            i += 1;
        }
    }
    return length;
}

idx_t sbd_distances_matrix(seq_t *matrix, idx_t nb_rows, idx_t nb_cols, seq_t* output, DTWBlock* block) {
    idx_t r, c, cb;
    idx_t length;
    idx_t i;
    seq_t value;
    
    length = dtw_distances_length(block, nb_rows);
    if (length == 0) {
        return 0;
    }
    
    // Correct block
    if (block->re == 0) {
        block->re = nb_rows;
    }
    if (block->ce == 0) {
        block->ce = nb_rows;
    }
    
    i = 0;
    for (r=block->rb; r<block->re; r++) {
        if (block->triu && r + 1 > block->cb) {
            cb = r+1;
        } else {
            cb = block->cb;
        }
        for (c=cb; c<block->ce; c++) {
            value = sbd_distance(&matrix[r*nb_cols], nb_cols, &matrix[c*nb_cols], nb_cols);
//            printf("i=%zu - r=%zu - c=%zu - value=%.4f\n", i, r, c, value);
            output[i] = value;
            i += 1;
        }
    }
    assert(length == i);
    return length;
}

idx_t sbd_distances_ptrs_parallel(seq_t **ptrs, idx_t nb_ptrs, idx_t* lengths, seq_t* output, DTWBlock* block) {
    idx_t r, c, r_i, c_i;
    idx_t length;
    idx_t *cbs, *rls;

    if (dtw_distances_prepare(block, nb_ptrs, &cbs, &rls, &length, NULL) != 0) {
        return 0;
    }
    
#if defined(_OPENMP)
    r_i=0;
    // Rows have different lengths, thus use guided scheduling to make threads with shorter rows
    // not wait for threads with longer rows. Also the first rows are always longer than the last
    // ones (upper triangular matrix), so this nicely aligns with the guided strategy.
    // Using schedule("static, 1") is also fast for the same reason (neighbor rows are almost
    // the same length, thus a circular assignment works well) but assumes all DTW computations take
    // the same amount of time.
    #pragma omp parallel for private(r_i, c_i, r, c) schedule(guided)
    for (r_i=0; r_i < (block->re - block->rb); r_i++) {
        r = block->rb + r_i;
        c_i = 0;
        if (block->triu) {
            c = cbs[r_i];
        } else {
            c = block->cb;
        }
        for (; c<block->ce; c++) {
            double value = sbd_distance(ptrs[r], lengths[r], ptrs[c], lengths[c]);
            if (block->triu) {
                output[rls[r_i] + c_i] = value;
            } else {
                output[(block->ce - block->cb) * r_i + c_i] = value;
            }
            c_i++;
        }
    }
    
    if (block->triu) {
        free(cbs);
        free(rls);
    }
    return length;
#else
    printf("ERROR: DTAIDistanceC is compiled without OpenMP support.\n");
    for  (r_i=0; r_i<length; r_i++) {
        output[r_i] = 0;
    }
    return 0;
#endif
}

idx_t sbd_distances_matrix_parallel(seq_t *matrix, idx_t nb_rows, idx_t nb_cols, seq_t* output, DTWBlock* block) {
    idx_t r, c, r_i, c_i;
    idx_t length;
    idx_t *cbs, *rls;

    if (dtw_distances_prepare(block, nb_rows, &cbs, &rls, &length, NULL) != 0) {
        return 0;
    }

#if defined(_OPENMP)
    r_i = 0;
    #pragma omp parallel for private(r_i, c_i, r, c) schedule(guided)
    for (r_i=0; r_i < (block->re - block->rb); r_i++) {
        r = block->rb + r_i;
        c_i = 0;
        if (block->triu) {
            c = cbs[r_i];
        } else {
            c = block->cb;
        }
        for (; c<block->ce; c++) {
            double value = sbd_distance(&matrix[r*nb_cols], nb_cols,
                                         &matrix[c*nb_cols], nb_cols);
            if (block->triu) {
                output[rls[r_i] + c_i] = value;
            } else {
                output[(block->ce - block->cb) * r_i + c_i] = value;
            }
            c_i++;
        }
    }
    
    if (block->triu) {
        free(cbs);
        free(rls);
    }
    return length;
#else
    printf("ERROR: DTAIDistanceC is compiled without OpenMP support.\n");
    for  (r_i=0; r_i<length; r_i++) {
        output[r_i] = 0;
    }
    return 0;
#endif
}
