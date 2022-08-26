#include <math.h>
#include "dd_dtw.h"
#include "dd_sbd.h"


void lower_upper_naive(seq_t *t, idx_t len, seq_t *l, seq_t *u, DTWSettings *settings);
void lower_upper_ascending_minima(seq_t *t, idx_t len, seq_t *l, seq_t *u, DTWSettings *settings);

seq_t lb_keogh_from_envelope(seq_t *s1, idx_t l1, seq_t *l, seq_t *u, DTWSettings *settings);

int nn_naive(int K, seq_t *query, idx_t query_size, seq_t *data, idx_t data_size, int verbose, idx_t *location, seq_t *distance, DTWSettings *settings);
int nn_sbd(int K, seq_t *query, idx_t query_size, seq_t *data, idx_t data_size, int verbose, idx_t *location, seq_t *distance);
int nn_ucrdtw(int K, seq_t *query, idx_t query_size, seq_t *l, seq_t *u, seq_t *data, idx_t data_size, int verbose, idx_t *location, seq_t *distance, DTWSettings *settings);
int nn_ucrdtw_data(int K, seq_t *query, idx_t query_size, seq_t *data, idx_t data_size, seq_t *l, seq_t *u, int verbose, idx_t *location, seq_t *distance, DTWSettings *settings);
