// quantization helper functions
#ifndef __QUANT_H__
#define __QUANT_H__
#include<math.h>
#include<stdint.h>

static int GS = 64; // group size global for quantization of the weights

typedef struct {
    int8_t* q;    // quantized values
    float* s; // scaling factors
} QuantizedTensor;

void dequantize(QuantizedTensor *qx, float* x, int n);
void quantize(QuantizedTensor *qx, float* x, int n);

#endif