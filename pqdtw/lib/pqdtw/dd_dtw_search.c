#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#include "dd_dtw_search.h"

#define min(x,y) ((x)<(y)?(x):(y))
#define max(x,y) ((x)>(y)?(x):(y))
#define dist(x,y) ((x-y)*(x-y))

#define INF 1e20 // Pseudo-infinite number for this code

/// Data structure for sorting the query
typedef struct index {
    double value;
    int index;
} index_t;

/// Sorting function for the query, sort by abs(z_norm(q[i])) from high to low
int index_comp(const void* a, const void* b) {
    index_t* x = (index_t*) a;
    index_t* y = (index_t*) b;
    double v = fabs(y->value) - fabs(x->value);   // high to low
	if (v < 0) return -1;
	if (v > 0) return 1;
	return 0;
}

struct pairs_double {
  double value;
  int death;
};

//TODO: not correct if psi_1 != psi_2


/*!
Compute LB_Kim between First and Last pair of points.

LB_Kim is the simplest lower bound for DTW with constant O(1) complexity. It
extracts four features: distances of the top, bottom, first and last points
from the sequence. The maximum of all four features is the lower bound for DTW.
However, when z-normalized, the top and bottom cannot give significant
benefits and these features are omitted.
The pruning power of LB_Kim is non-trivial, especially when the query is not
long, say in length 128.

@param t First sequence.
@param q Second sequence.
@param len Length of the sequences.
@param settings A DTWSettings struct with options for the DTW algorithm.
*/
seq_t lb_kim_fl(seq_t *t, seq_t *q, idx_t len, DTWSettings *settings) {
    idx_t start = settings->psi_1b;
    idx_t end = len - settings->psi_1e;

    seq_t best_so_far = settings->max_dist;
    if (best_so_far == 0) {
        best_so_far = INF;
    } else {
        best_so_far = pow(best_so_far, 2);
    }


    /// 1 point at front and back
    seq_t d, lb;
    seq_t x0 = t[start];
    seq_t y0 = t[end - 1];
    lb = dist(x0, q[start]) + dist(y0, q[end - 1]);
    if (lb >= best_so_far)
        return sqrt(lb);

    /// 2 points at front
    seq_t x1 = t[start + 1];
    d = min(dist(x1, q[start]), dist(x0, q[start + 1]));
    d = min(d, dist(x1, q[start + 1]));
    lb += d;
    if (lb >= best_so_far)
        return sqrt(lb);

    /// 2 points at back
    seq_t y1 = t[end - 2];
    d = min(dist(y1, q[end - 1]), dist(y0, q[end - 2]));
    d = min(d, dist(y1, q[end - 2]));
    lb += d;
    if (lb >= best_so_far)
        return sqrt(lb);

    /// 3 points at front
    double x2 = t[start + 2];
    d = min(dist(x0, q[start + 2]), dist(x1, q[start + 2]));
    d = min(d, dist(x2, q[start + 2]));
    d = min(d, dist(x2, q[start + 1]));
    d = min(d, dist(x2, q[start + 0]));
    lb += d;
    if (lb >= best_so_far)
        return sqrt(lb);

    /// 3 points at back
    double y2 = t[end - 3];
    d = min(dist(y0, q[end - 3]), dist(y1, q[end - 3]));
    d = min(d, dist(y2,q[end - 3]));
    d = min(d, dist(y2,q[end - 2]));
    d = min(d, dist(y2,q[end - 1]));
    lb += d;

    return sqrt(lb);
}

/*!
Find the envelope for LB_Keogh.

The upper and lower bounds are computed as the minimum and maximum value in 
a window around each sample as large as the Sakoe-Chiba band. The implmentation
uses the Ascending Minima algorithm.

@param t Sequence for which the lower bound will be computed.
@param len Length of the sequence.
@param l Empty array of length `len` in which the lower envelope will be stored.
@param u Empty array of length `len` in which the upper envelope will be stored.
@param settings A DTWSettings struct with options for the DTW algorithm.
*/
void lower_upper_ascending_minima(seq_t *t, idx_t len, seq_t *l, seq_t *h, DTWSettings *settings) {
  idx_t window = settings->window;
  if (window == 0 || window > len - 1) {
    window = len - 1;
  }
  int i,ii;
  int kk = window*2+1;
  if(len<1) printf("ERROR: n must be > 0\n");

  /* structs  */
  struct pairs_double * ring_l;
  struct pairs_double * minpair;
  struct pairs_double * end_l;
  struct pairs_double * last_l;

  struct pairs_double * ring_h;
  struct pairs_double * maxpair;
  struct pairs_double * end_h;
  struct pairs_double * last_h;


  /* init l env */
  ring_l = malloc(kk * sizeof *ring_l);
  if (!ring_l) printf("ERROR: malloc error\n");
  end_l  = ring_l + kk;
  last_l = ring_l;
  minpair = ring_l;
  minpair->value = t[0];
  minpair->death = kk;


  /* init upper env */
  ring_h = malloc(kk * sizeof *ring_h);
  if (!ring_h) printf("ERROR: malloc error\n");
  end_h  = ring_h + kk;
  last_h = ring_h;
  maxpair = ring_h;
  maxpair->value = t[0];
  maxpair->death = kk;

  /* start and main window  */
  ii = 0;
  for (i=1;i<=len+window;i++) {
    if(ii<len-1) ii++;
    if(i>window){
      l[i-window-1] = minpair->value;
      h[i-window-1] = maxpair->value;
    }

    /* lower */
    if (minpair->death == i) {
      minpair++;
      if (minpair >= end_l) minpair = ring_l;
    }
    if (t[ii] <= minpair->value) {
      minpair->value = t[ii];
      minpair->death = i+kk;
      last_l = minpair;
    } else {
      while (last_l->value >= t[ii]) {
        if (last_l == ring_l) last_l = end_l;
        --last_l;
      }
      ++last_l;
      if (last_l == end_l) last_l = ring_l;
      last_l->value = t[ii];
      last_l->death = i+kk;
    }

    /* upper */
    if (maxpair->death == i) {
      maxpair++;
      if (maxpair >= end_h) maxpair = ring_h;
    }
    if (t[ii] >= maxpair->value) {
      maxpair->value = t[ii];
      maxpair->death = i+kk;
      last_h = maxpair;
    } else {
      while (last_h->value <= t[ii]) {
        if (last_h == ring_h) last_h = end_h;
        --last_h;
      }
      ++last_h;
      if (last_h == end_h) last_h = ring_h;
      last_h->value = t[ii];
      last_h->death = i+kk;
    }
  }
  free(ring_l);
  free(ring_h);
}

/*!
Find the envelope for LB_Keogh.

The upper and lower bounds are computed as the minimum and maximum value in 
a window around each sample as large as the Sakoe-Chiba band. This is a naive
implmentation that serves as a benchmark for lower_upper_ascending_minima.

@param t Sequence for which the evelope will be computed.
@param len Length of the sequence.
@param l Empty array of length `len` in which the lower envelope will be stored.
@param u Empty array of length `len` in which the upper envelope will be stored.
@param settings A DTWSettings struct with options for the DTW algorithm.
*/
void lower_upper_naive(seq_t *t, idx_t len, seq_t *l, seq_t *u, DTWSettings *settings) {
    idx_t window = settings->window;
    if (window == 0) {
        window = len;
    }
    idx_t imin, imax;
    for (idx_t i=0; i<len; i++) {
        imin = max(0, i - window);
        imax = min(i + window + 1, len);
        u[i] = 0;
        l[i] = INF;
        for (idx_t j=imin; j<imax; j++) {
            if (t[j] > u[i]) {
                u[i] = t[j];
            }
            if (t[j] < l[i]) {
                l[i] = t[j];
            }
        }
    }
}


/*!
Compute the Keogh lower bound for DTW with precomputed bounds.

LB_Keogh is calculated as the Euclidean distance between the observations of
s that fall outside the envelope and the nearest upper or lower sequence. Note
that s and (l, u) should have equal lengths.

@param s Sequence for which the lower bound will be computed.
@param len Length of the sequence.
@param l The lower envelope of the second sequence.
@param u The upper envelpe of the second sequence.
@param settings A DTWSettings struct with options for the DTW algorithm.
 */
seq_t lb_keogh_from_envelope(seq_t *s, idx_t len, seq_t *l, seq_t *u, DTWSettings *settings) {
    idx_t lb = 0;
    seq_t ci;
    seq_t best_so_far = settings->max_dist;
    if (best_so_far == 0) {
        best_so_far = INF;
    } else {
        best_so_far = pow(best_so_far, 2);
    }

    for (idx_t i=settings->psi_1b; (i < len - settings->psi_1e) && (lb < best_so_far); i++) {
        ci = s[i];
        if (ci > u[i]) {
            lb += dist(ci, u[i]);
        } else if (ci < l[i]) {
            lb += dist(l[i], ci);
        }
    }
    return sqrt(lb);
}

/*!
Compute the Keogh lower bound for DTW with precomputed bounds on a sorted query.

LB_Keogh is calculated as the Euclidean distance between the observations of
s that fall outside the envelope and the nearest upper or lower sequence. 

Instead of computing the LB from left to right, this implementation supports
a custom ordering of points. This leads to earlier abonding. Keogh et al.
recommend ordering s by the absolute height of the query point. 

Note that s and (l, u) should have equal lengths.

@param order Sorted indices for the query s.
@param s Sequence for which the lower bound will be computed.
@param len Length of the sequence.
@param l The ordered lower envelope of the second sequence.
@param u The ordered upper envelpe of the second sequence.
@param settings A DTWSettings struct with options for the DTW algorithm.
 */
seq_t lb_keogh_from_query_envelope_sorted(idx_t* order, seq_t *s, idx_t len, seq_t *l, seq_t *u, DTWSettings *settings) {
    idx_t lb = 0;
    seq_t ci, ui, li;
    seq_t best_so_far = settings->max_dist;
    if (best_so_far == 0) {
        best_so_far = INF;
    } else {
        best_so_far = pow(best_so_far, 2);
    }

    for (idx_t i=settings->psi_1b; (i < len - settings->psi_1e) && (lb < best_so_far); i++) {
        ci = s[order[i]];
        ui = u[i];
        li = l[i];
        if (ci > ui) {
            lb += dist(ci, ui);
        } else if (ci < li) {
            lb += dist(li, ci);
        }
    }
    return sqrt(lb);
}


/*!
Compute the Keogh lower bound for DTW with precomputed bounds on a sorted query.

LB_Keogh is calculated as the Euclidean distance between the observations of
s that fall outside the envelope and the nearest upper or lower sequence. 

Instead of computing the LB from left to right, this implementation supports
a custom ordering of points. This leads to earlier abonding. Keogh et al.
recommend ordering s by the absolute height of the query point. 

Note that s and (l, u) should have equal lengths.

@param order Sorted indices for the query s.
@param s Ordered sequence for which the lower bound will be computed.
@param len Length of the sequence.
@param l The lower envelope of the second sequence.
@param u The upper envelpe of the second sequence.
@param settings A DTWSettings struct with options for the DTW algorithm.
 */
seq_t lb_keogh_from_data_envelope_sorted(idx_t* order, seq_t *s, idx_t len, seq_t *l, seq_t *u, DTWSettings *settings) {
    idx_t lb = 0;
    seq_t ci, ui, li;
    seq_t best_so_far = settings->max_dist;
    if (best_so_far == 0) {
        best_so_far = INF;
    } else {
        best_so_far = pow(best_so_far, 2);
    }
    
    for (idx_t i=settings->psi_1b; (i < len - settings->psi_1e) && (lb < best_so_far); i++) {
        ci = s[i];
        ui = u[order[i]];
        li = l[order[i]];
        if (ci > ui) {
            lb += dist(ci, ui);
        } else if (ci < li) {
            lb += dist(li, ci);
        }
    }
    return sqrt(lb);
}

/*!
Helper for KNN with K > 1.
*/
static double set_knn(seq_t dist, seq_t *_kvec, idx_t *_lvec, idx_t *wk, idx_t loc, int K) {
  seq_t bsf = 0.0;
  _kvec[*wk] = dist;
  _lvec[*wk] = loc;
  /* Now for K-NN redefine bsf to the "worst of the" "best so far" and save it's index at wk */
  for(int k=0; k<K; k++) { 
    if(_kvec[k] > bsf) {
      bsf = _kvec[k];
      *wk  = k;
    }
  }
  return bsf;
}

/*!
Find the KNN of a query using DTW.

This is a naive implementation that exhaustively computes the DTW between the
query sequence and each sequence in the database.

The query and all sequences in the database should have equal lengths.

@param K The number of neighbors to find.
@param data The database that will be searched.
@param data_size The number of sequences in the database.
@param query The query sequence.
@param query_size The lenght of the query and each sequence in the database.
@param verbose Whether to print some performance stats.
@param location Index to the kNN.
@param distance DTW distance between the KNN and the query.
@param settings A DTWSettings struct with options for the DTW algorithm.
*/
int nn_naive(int K, seq_t *query, idx_t query_size, seq_t *data, idx_t data_size, int verbose, idx_t *location, seq_t *distance, DTWSettings *settings) {
    seq_t bsf = INF;
    idx_t di;
    seq_t score = 0;
    DTWSettings cur_settings = *settings;
    if (cur_settings.max_dist == 0) {
        cur_settings.max_dist = INF;
    }

    int k;
    if(K < 1) K = 1;
    idx_t wk = 0;  /* index position of worst of the KNN */
    
    double t1 = 0, t2;
    if (verbose) {
        t1 = clock();
    }
    
    for (k = 0; k < K; k++) {
      distance[k] = INF;
      location[k] = 0;
    }


    for (di=0; di<data_size; di++) {
        score = dtw_distance(&data[di*query_size], query_size, query, query_size, &cur_settings);
        if (score < bsf) {
            if(K==1) {
                distance[0] = score;
                location[0] = di;
                bsf      = score;
            } else {
                bsf = set_knn(score, distance, location, &wk, di, K);
            }
            cur_settings.max_dist = bsf;
        }
    }

    if (verbose) {
        t2 = clock();
        printf("Location : %ld\n", location[0]);
        printf("Distance : %.6f\n", bsf);
        printf("Data Scanned : %ld\n", di);
        printf("Total Execution Time : %.4f secs\n", (t2 - t1) / CLOCKS_PER_SEC);
        printf("\n");
    }
    return 0;
}

/*!
Find the KNN of a query using SBD.

The query and all sequences in the database should have equal lengths.

@param K The number of neighbors to find.
@param data The database that will be searched.
@param data_size The number of sequences in the database.
@param query The query sequence.
@param query_size The lenght of the query and each sequence in the database.
@param verbose Whether to print some performance stats.
@param location Index to the kNN.
@param distance SBD distance between the KNN and the query.
*/
int nn_sbd(int K, seq_t *query, idx_t query_size, seq_t *data, idx_t data_size, int verbose, idx_t *location, seq_t *distance) {
    seq_t bsf = INF;
    idx_t di;
    seq_t score = 0;

    int k;
    if(K < 1) K = 1;
    idx_t wk = 0;  /* index position of worst of the KNN */
    
    double t1 = 0, t2;
    if (verbose) {
        t1 = clock();
    }
    
    for (k = 0; k < K; k++) {
      distance[k] = INF;
      location[k] = 0;
    }


    for (di=0; di<data_size; di++) {
        score = sbd_distance(&data[di*query_size], query_size, query, query_size);
        if (score < bsf) {
            if(K==1) {
                distance[0] = score;
                location[0] = di;
                bsf      = score;
            } else {
                bsf = set_knn(score, distance, location, &wk, di, K);
            }
        }
    }

    if (verbose) {
        t2 = clock();
        printf("Location : %ld\n", location[0]);
        printf("Distance : %.6f\n", bsf);
        printf("Data Scanned : %ld\n", di);
        printf("Total Execution Time : %.4f secs\n", (t2 - t1) / CLOCKS_PER_SEC);
        printf("\n");
    }
    return 0;
}

/*!
Find the 1NN of a query using DTW.

This is a fast implementation that uses the UCRDTW optimizations to prune DTW
computations. 

The query and all sequences in the database should have equal lengths.

@param query The query sequence.
@param query_size The lenght of the query and each sequence in the database.
@param l The lower envelope of the query sequence.
@param u The upper envelpe of the query sequence.
@param data The database that will be searched.
@param data_size The number of sequences in the database.
@param verbose Whether to print some performance stats.
@param location Index to the 1NN.
@param distance DTW distance between the 1NN and the query.
@param settings A DTWSettings struct with options for the DTW algorithm.
*/
int nn_ucrdtw(int K, seq_t *query, idx_t query_size, seq_t *l, seq_t *u, seq_t *data, idx_t data_size, int verbose, idx_t *location, seq_t *distance, DTWSettings *settings) {
    seq_t bsf = INF;
    idx_t di;
    seq_t score = 0;
    seq_t lb_kim, lb_keogh, lb_keogh2;
    DTWSettings cur_settings = *settings;
    if (cur_settings.max_dist == 0) {
        cur_settings.max_dist = INF;
    }

    idx_t *order; ///new order of the query
    seq_t *qo, *uo, *lo;
    index_t *q_tmp;
    seq_t *u_buff, *l_buff;

    int k;
    if(K < 1) K = 1;
    idx_t wk = 0;  /* index position of worst of the KNN */

    for (k = 0; k < K; k++) {
      distance[k] = INF;
      location[k] = 0;
    }

    idx_t i = 0;
    idx_t kim = 0;
    idx_t keogh = 0;
    idx_t keogh2 = 0;
    double t1 = 0, t2;
    if (verbose) {
        t1 = clock();
    }

    qo = (seq_t*) calloc(query_size, sizeof(seq_t));
    if (qo == NULL) {
        printf("ERROR: Memory can't be allocated!\n");
        return -1;
    }

    uo = (seq_t*) calloc(query_size, sizeof(seq_t));
    if (uo == NULL) {
        printf("ERROR: Memory can't be allocated!\n");
        return -1;
    }

    lo = (seq_t*) calloc(query_size, sizeof(seq_t));
    if (lo == NULL) {
        printf("ERROR: Memory can't be allocated!\n");
        return -1;
    }

    order = (idx_t *) calloc(query_size, sizeof(idx_t));
    if (order == NULL) {
        printf("ERROR: Memory can't be allocated!\n");
        return -1;
    }

    q_tmp = (index_t *) calloc(query_size, sizeof(index_t));
    if (q_tmp == NULL) {
        printf("ERROR: Memory can't be allocated!\n");
        return -1;
    }

    u_buff = (seq_t*) calloc(query_size, sizeof(seq_t));
    if (u_buff == NULL) {
        printf("ERROR: Memory can't be allocated!\n");
        return -1;
    }

    l_buff = (seq_t*) calloc(query_size, sizeof(seq_t));
    if (l_buff == NULL) {
        printf("ERROR: Memory can't be allocated!\n");
        return -1;
    }

    // Sort the query one time by abs(z-norm(q[i]))
    for (i = 0; i < query_size; i++) {
        q_tmp[i].value = query[i];
        q_tmp[i].index = i;
    }
    qsort(q_tmp, query_size, sizeof(index_t), index_comp);

    // also create another arrays for keeping sorted envelope
    for (i = 0; i < query_size; i++) {
        int o = q_tmp[i].index;
        order[i] = o;
        qo[i] = query[o];
        uo[i] = u[o];
        lo[i] = l[o];
    }

    for (di=0; di<data_size; di++) {
        // Use a constant lower bound to prune the obvious subsequence
        lb_kim = lb_kim_fl(&data[di*query_size], query, query_size, &cur_settings);
        if (lb_kim < bsf) {
            // Use the Keogh lower bound with a precomputed evelope around the query
            lb_keogh = lb_keogh_from_query_envelope_sorted(order, &data[di*query_size], query_size, lo, uo, &cur_settings);
            if (lb_keogh < bsf) {
                lower_upper_ascending_minima(&data[di*query_size], query_size, l_buff, u_buff, &cur_settings);
                lb_keogh2 = lb_keogh_from_data_envelope_sorted(order, qo, query_size, l_buff, u_buff, &cur_settings);
                if (lb_keogh2 < bsf) {
                    score = dtw_distance(&data[di*query_size], query_size, query, query_size, &cur_settings);
                    if (score < bsf) {
                      if(K==1) {
                        distance[0] = score;
                        location[0] = di;
                        bsf      = score;
                      } else {
                        bsf = set_knn(score, distance, location, &wk, di, K);
                      }
                      cur_settings.max_dist = bsf;
                    }
                } else {
                  keogh2++;
                }
            } else {
              keogh++;
            }
        }
    }

    if (verbose) {
        t2 = clock();
        printf("Location : %ld\n", location[0]);
        printf("Distance : %.6f\n", bsf);
        printf("Data Scanned : %ld\n", di);
        printf("Total Execution Time : %.4f secs\n", (t2 - t1) / CLOCKS_PER_SEC);
        printf("\n");
        printf("Pruned by LB_Kim  : %6.2f%%\n", ((double) kim / di) * 100);
        printf("Pruned by LB_Keogh  : %6.2f%%\n", ((double) keogh / di) * 100);
        printf("Pruned by LB_Keogh 2 : %6.2f%%\n", ((double) keogh2 / di) * 100);
        printf("DTW Calculation     : %6.2f%%\n", 100 - (((double) kim + keogh + keogh2) / di * 100));
    }

    free(qo);
    free(uo);
    free(lo);
    free(order);
    free(q_tmp);
    free(l_buff);
    free(u_buff);

    return 0;
}

/*!
Find the 1NN of a query using DTW.

This is a fast implementation that uses the UCRDTW optimizations to prune DTW
computations. In contrast to `nn_lb_keogh`, the keogh bounds are precomputed
for each sequence in the database.

The query and all sequences in the database should have equal lengths.

@param query The query sequence.
@param query_size The lenght of the query and each sequence in the database.
@param data The database that will be searched.
@param data_size The number of sequences in the database.
@param l The lower envelope of each database sequence.
@param u The upper envelpe of each database sequence.
@param verbose Whether to print some performance stats.
@param location Index to the 1NN.
@param distance DTW distance between the 1NN and the query.
@param settings A DTWSettings struct with options for the DTW algorithm.
*/
int nn_ucrdtw_data(int K, seq_t *query, idx_t query_size, seq_t *data, idx_t data_size, seq_t *l, seq_t *u, int verbose, idx_t *location, seq_t *distance, DTWSettings *settings) {
    seq_t bsf = INF;
    idx_t di;
    seq_t score = 0;
    index_t *q_tmp;
    idx_t *order; ///new order of the query
    seq_t *qo;
    DTWSettings cur_settings = *settings;
    if (cur_settings.max_dist == 0) {
        cur_settings.max_dist = INF;
    }

    seq_t lb_kim, lb_keogh;

    int k;
    if(K < 1) K = 1;
    idx_t wk = 0;  /* index position of worst of the KNN */

    for (k = 0; k < K; k++) {
      distance[k] = INF;
      location[k] = 0;
    }

    idx_t i = 0;
    int kim = 0;
    int keogh = 0;
    double t1 = 0, t2;
    if (verbose) {
        t1 = clock();
    }


    qo = (seq_t*) calloc(query_size, sizeof(seq_t));
    if (qo == NULL) {
        printf("ERROR: Memory can't be allocated!\n");
        return -1;
    }

    order = (idx_t*) calloc(query_size, sizeof(idx_t));
    if (order == NULL) {
        printf("ERROR: Memory can't be allocated!\n");
        return -1;
    }

    q_tmp = (index_t *) calloc(query_size, sizeof(index_t));
    if (q_tmp == NULL) {
        printf("ERROR: Memory can't be allocated!\n");
        return -1;
    }

    // Sort the query one time by abs(z-norm(q[i]))
    for (i = 0; i < query_size; i++) {
        q_tmp[i].value = query[i];
        q_tmp[i].index = i;
    }
    qsort(q_tmp, query_size, sizeof(index_t), index_comp);
    // also create another arrays for keeping sorted envelope
    idx_t o;
    for (i = 0; i < query_size; i++) {
        o = q_tmp[i].index;
        order[i] = o;
        qo[i] = query[o];
    }

    for (di=0; di < data_size; di++) {
      // Use a constant lower bound to prune the obvious subsequence
      lb_kim = lb_kim_fl(&data[di*query_size], query, query_size, &cur_settings);
      if (lb_kim < bsf) {
        // Use the Keogh lower bound with a precomputed evelope around the data
        lb_keogh = lb_keogh_from_data_envelope_sorted(order, qo, query_size, &l[di*query_size], &u[di*query_size], &cur_settings);
        if (lb_keogh < bsf) {
          // Compute the DTW distance with early stopping
          score = dtw_distance(&data[di*query_size], query_size, query, query_size, &cur_settings);
          if (score < bsf) {
            if(K==1) {
              distance[0] = score;
              location[0] = di;
              bsf         = score;
            } else {
              bsf = set_knn(score, distance, location, &wk, di, K);
            }
            cur_settings.max_dist = bsf;
          }
        } else {
          keogh++;
        }
      } else {
          kim++;
      }
    }

    if (verbose) {
        t2 = clock();
        printf("Location : %ld\n", location[0]);
        printf("Distance : %.6f\n", bsf);
        printf("Data Scanned : %ld\n", di);
        printf("Total Execution Time : %.4f secs\n", (t2 - t1) / CLOCKS_PER_SEC);
        printf("\n");
        printf("Pruned by LB_Kim  : %6.2f%%\n", ((double) kim / di) * 100);
        printf("Pruned by LB_Keogh  : %6.2f%%\n", ((double) keogh / di) * 100);
        printf("DTW Calculation     : %6.2f%%\n", 100 - (((double) keogh + kim) / di * 100));
    }

    free(qo);
    free(order);
    free(q_tmp);

    return 0;
}
