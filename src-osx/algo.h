#ifndef __ALGOLIB_OSX_H__
#define __ALGOLIB_OSX_H__
#include <stdint.h>
#include <math.h>

#include "quant.h"

// --- neural net blocks ---
// from orig repo

void rmsnorm(float* o, float* x, float* weight, int size);

void matmulf(float* xout, float* x, float* w, int n, int d);
void matmulq(float* xout, QuantizedTensor *x, QuantizedTensor *w, int n, int d);

void softmax(float* x, int size);

#endif