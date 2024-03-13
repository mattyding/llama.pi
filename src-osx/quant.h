// quantization helper functions
#ifndef __QUANT_OSX_H__
#define __QUANT_OSX_H__

#include<stdint.h>
#include <stdlib.h>
#include<math.h>

static int GS = 64; // group size global for quantization of the weights

typedef struct {
    int8_t* q;    // quantized values
    float* s; // scaling factors
} QuantizedTensor;

void dequantize(QuantizedTensor *qx, float* x, int n);
void quantize(QuantizedTensor *qx, float* x, int n);

QuantizedTensor *init_quantized_tensors(void **ptr, int n, int size_each);

#endif