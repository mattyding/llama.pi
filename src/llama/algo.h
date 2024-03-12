#ifndef __ALGOLIB_H__
#define __ALGOLIB_H__
#include <stdint.h>
#include <math.h>

#include "rpi.h"
#include "quant.h"

// --- neural net blocks ---
// from orig repo

void rmsnorm(float* o, float* x, float* weight, int size);

void matmulf(float* xout, float* x, float* w, int n, int d);
void matmulq(float* xout, QuantizedTensor *x, QuantizedTensor *w, int n, int d);

// called in sampler.c::sample
void softmax(float* x, int size);

// --- other stdlib algos ---

// tldr: I couldn't figure out compiler errors so I'm importing stdlib anyways but bsearch/qsort are special in that their definitions can't be found so i'm redefining them here.

// called in tokenizer.c::str_lookup
// src: https://github.com/gcc-mirror/gcc/blob/master/libiberty/bsearch.c
void * my_bsearch (const void *key, const void *base0,
         size_t nmemb, size_t size,
         int (*compar)(const void *, const void *));

// basic sorting function implementation
// qsort import fails ("undefined reference to `__aeabi_uidiv'") so here is a suboptimal but working algo
// called in tokenizer.c::encode and sampler.c::sample_topp
void my_qsort(void *base, size_t nmemb, size_t size, int (*compar)(const void *, const void *));

#endif