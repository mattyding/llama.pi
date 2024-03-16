#ifndef __PRUNE_H__
#define __PRUNE_H__
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "quant.h"

// pruning experiments
static int use_prune = 1;   // 1 if using pruned weights
// static char *prune_path = "../models/7b-q80/prune/bitvec_l1_unstructured_0.2.bin"; // buggy
static float prune_ratio = 0.3;    // L1 pruning: percentage to prune

void prune_matrix(float* matrix, int size, float prune_ratio);
void prune_qmatrix(QuantizedTensor* qmatrix, int n, int size_each, float prune_ratio);

#endif