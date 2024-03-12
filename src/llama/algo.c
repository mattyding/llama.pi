#include "algo.h"

void rmsnorm(float* o, float* x, float* weight, int size) {
    // calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    // normalize and scale
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

void matmulf(float* xout, float* x, float* w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    int i;
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}


void matmulq(float* xout, QuantizedTensor *x, QuantizedTensor *w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    // inputs to this function are both quantized

    int i;
    for (i = 0; i < d; i++) {

        float val = 0.0f;
        int32_t ival = 0;
        int in = i * n;

        // do the matmul in groups of GS
        int j;
        for (j = 0; j <= n - GS; j += GS) {
            for (int k = 0; k < GS; k++) {
                ival += ((int32_t) x->q[j + k]) * ((int32_t) w->q[in + j + k]);
            }
            val += ((float) ival) * w->s[(in + j) / GS] * x->s[j / GS];
            ival = 0;
        }

        xout[i] = val;
    }
}

void softmax(float* x, int size) {
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

// --- other stdlib algos ---

void * my_bsearch (const void *key, const void *base0,
         size_t nmemb, size_t size,
         int (*compar)(const void *, const void *)) {
	const char *base = (const char *) base0;
	int lim, cmp;
	const void *p;

	for (lim = nmemb; lim != 0; lim >>= 1) {
		p = base + (lim >> 1) * size;
		cmp = (*compar)(key, p);
		if (cmp == 0)
			return (void *)p;
		if (cmp > 0) {	/* key > p: move right */
			base = (const char *)p + size;
			lim--;
		} /* else move left */
	}
	return NULL;
}

#if 0
void qsort(void *base, size_t nmemb, size_t size, int (*compar)(const void *, const void *)) {
    // personally avoiding kmalloc calls b/c can't free
    // can potentially lead to horrible memory corruption. unsure... we will see...
    char *arr = (char *)base;
    size_t i, j, min_idx;
    char temp;

    for (i = 0; i < nmemb - 1; i++) {
        min_idx = i;
        for (j = i + 1; j < nmemb; j++) {
            if (compar(arr + j * size, arr + min_idx * size) < 0) {
                min_idx = j;
            }
        }
        if (min_idx != i) {
            // Swap elements at i and min_idx
            for (j = 0; j < size; j++) {
                temp = arr[i * size + j];
                arr[i * size + j] = arr[min_idx * size + j];
                arr[min_idx * size + j] = temp;
            }
        }
    }
}
#endif

// backup qsort with malloc
void my_qsort(void *base, size_t nmemb, size_t size, int (*compar)(const void *, const void *)) {
    char *arr = (char *)base;
    char *temp = kmalloc(size);
    for (size_t i = 0; i < nmemb - 1; i++) {
        for (size_t j = i + 1; j < nmemb; j++) {
            if (compar(arr + i * size, arr + j * size) > 0) {
                memcpy(temp, arr + i * size, size);
                memcpy(arr + i * size, arr + j * size, size);
                memcpy(arr + j * size, temp, size);
            }
        }
    }
}