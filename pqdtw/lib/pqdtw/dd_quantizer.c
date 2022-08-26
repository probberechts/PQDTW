#include <stdio.h>
#include <string.h>
#include <math.h>

#include "../../wavelib/header/wavelib.h"
#include "dd_quantizer.h"


void zscore_normalization(seq_t *values, idx_t length) {
    seq_t mean, std;
    
    // Computation of standard deviation by sample formula
    seq_t differrence_sum;
    
    // calculate arithmetic mean 
    mean = 0;
    for(idx_t i=0; i < length; i++){
        mean += values[i];
    }
    mean /= length;

    // calculate sum (xi - mean)^2
    differrence_sum = 0;
    for(idx_t i=0; i < length; i++){
        differrence_sum += pow(values[i] - mean, 2);
    }
    // divide sum by N - 1 
    differrence_sum /= (length - 1); // this is sample std deviation. Remove the -1 to make it population std_dev
    // take the sqrt
    std = sqrt(differrence_sum);

    // special case, when the vector is a straight line and has no variance
    // the std results in zero, all values in the vector are set to zero
    if(std == 0) {   
        memset(values, 0, length * sizeof(*values));
    } else {
        // calculate the z score for each element
        for(idx_t i=0; i < length; i++) {
            values[i] = (values[i] - mean) / std;
        }
    }

}

/*!
Split a series in subspaces, using MODWT to dermine the split points.

First, the Maximal Overlap Discrete Wavelet Transform (MODWT) is used to identify
local structures in the provided time series. The boundaries of these local structures 
are extracted as the points at which the signs of the differences between the
time series data and MODWT scale coefficients change. Second, the time series is split in 
equal-lenght partitions. If the tail of such a partition contains a MODWT boundary, 
this boundary is used as the split point instead.

@param s 1-dimensional array containing the time series
@param l Length of the time series
@param J Level of the MODWT transform
@param subspace_size Size of the subspaces
@param tail Max lenght of the tail
@param out 1-dimensional array containing the split points.
*/
int get_wavelet_splits(seq_t *s, idx_t l, int J, idx_t subspace_size, idx_t tail, seq_t *out) {

    // Wavelet transform
    wave_object obj;
    wt_object wt;

    char *name = "haar";
    obj = wave_init(name);

    wt = wt_init(obj, "modwt", l, J);// Initialize the wavelet transform object

    modwt(wt, s);// Perform MODWT

    wave_free(obj);

    // Get wavelet split indices
    idx_t* wt_idx;
    int n_idx = 0;
    idx_t i, j;
    seq_t a, b;
    wt_idx = (idx_t*)malloc(sizeof(idx_t)* l);
    for (i=0; i < l-1; i++) {
      a = s[i] - wt->output[i];
      b = s[i+1] - wt->output[i+1];
      if ((a > 0) - (b < 0) == 0) {// Detect sign switch between original signal and scaling coef
        wt_idx[n_idx] = i - (1 << (J - 1));// Correct index with modwt phase shift
        n_idx++;
      }
    }

    wt_free(wt);

    // Get subspace split indices
    idx_t* split_idx;
    idx_t nb_subspaces = l / subspace_size;
    split_idx = (idx_t*)malloc(sizeof(idx_t)* (nb_subspaces+1));
    idx_t idx_ptr = 0;
    split_idx[0] = 0;
    for (i=1; i < nb_subspaces; i++) {
      split_idx[i] = i*subspace_size;
      while (idx_ptr < n_idx && wt_idx[idx_ptr] < i*subspace_size) {
        if (wt_idx[idx_ptr] > i*subspace_size-tail) {
           split_idx[i] = wt_idx[idx_ptr];
        }
        idx_ptr++;
      }
    }
    split_idx[i] = l+1;

    // v2
    idx_t l_subspace = subspace_size + tail;
    float l_diff;
    for (i=0; i < nb_subspaces; i++) {
      l_diff = (split_idx[i+1] - split_idx[i] - 1) / (float) l_subspace;
      for (j=0; j < l_subspace; j++) {
        out[i*l_subspace+j] = s[split_idx[i] + (idx_t)(j*l_diff)];
        // out[i*l_subspace+j] = (s[split_idx[i] + (idx_t)floor(j*l_diff)] + s[split_idx[i] + (idx_t)ceil(j*l_diff)]) / 2;
      }
      zscore_normalization(&out[i*l_subspace], l_subspace);
    }

    free(wt_idx);
    free(split_idx);

    return 0;
}

/*!
Encode a time series.

@param s 1-dimensional array containing the time series
@param l Length of the time series
@param M The number of subspaces.
@param codes The encoding of s.
*/
int encode(seq_t *codebook, idx_t codebook_size, seq_t *lenv, seq_t *uenv, seq_t *s, idx_t l, int J, idx_t subspace_size, idx_t tail, int use_lb_keogh, DTWSettings *settings, idx_t *codes) {
  int M = l / subspace_size;
  seq_t* subspaces = (seq_t*)malloc(sizeof(seq_t)* (M * (subspace_size + tail)));
  int m;
  idx_t real_subspace_size = subspace_size + tail;
  seq_t distance;

  get_wavelet_splits(s, l, J, subspace_size, tail, subspaces);
  for (m=0; m < M; m++) {

    // Find the 1NN in the codebook
    if (use_lb_keogh) {
      nn_ucrdtw_data(
          1, &subspaces[m*real_subspace_size], real_subspace_size, 
          &codebook[m*codebook_size*real_subspace_size], codebook_size, 
          &lenv[m*codebook_size*real_subspace_size], 
          &uenv[m*codebook_size*real_subspace_size], 
          0, &codes[m], &distance, settings);
    } else {
      nn_naive(
          1, &subspaces[m*real_subspace_size], real_subspace_size,
          &codebook[m*codebook_size*real_subspace_size], codebook_size, 
          0, &codes[m], &distance, settings);
    }
  }

  free(subspaces);

  return 0;
}

int encode_overlap(seq_t *codebook, idx_t codebook_size, seq_t *lenv, seq_t *uenv, seq_t *s, idx_t l, idx_t subspace_size, int overlap, int use_lb_keogh, DTWSettings *settings, idx_t *codes) {
  int M = l / (subspace_size - overlap);
  int m;
  seq_t distance;

  for (m=0; m < M; m++) {

    // Find the 1NN in the codebook
    if (use_lb_keogh) {
      nn_ucrdtw_data(
          1, &s[m*(subspace_size - overlap)], subspace_size, 
          &codebook[m*codebook_size*subspace_size], codebook_size, 
          &lenv[m*codebook_size*subspace_size], 
          &uenv[m*codebook_size*subspace_size], 
          0, &codes[m], &distance, settings);
    } else {
      nn_naive(
          1, &s[m*(subspace_size - overlap)], subspace_size,
          &codebook[m*codebook_size*subspace_size], codebook_size, 
          0, &codes[m], &distance, settings);
    }
  }

  return 0;
}

int encode_sbd(seq_t *codebook, idx_t codebook_size, seq_t *s, idx_t l, int J, idx_t subspace_size, idx_t tail, idx_t *codes) {
  int M = l / subspace_size;
  seq_t* subspaces = (seq_t*)malloc(sizeof(seq_t)* (M * (subspace_size + tail)));
  int m;
  idx_t real_subspace_size = subspace_size + tail;
  seq_t distance;

  get_wavelet_splits(s, l, J, subspace_size, tail, subspaces);
  for (m=0; m < M; m++) {

    // Find the 1NN in the codebook
    nn_sbd(
        1, &subspaces[m*real_subspace_size], real_subspace_size,
        &codebook[m*codebook_size*real_subspace_size], codebook_size, 
        0, &codes[m], &distance);
  }

  free(subspaces);

  return 0;
}
