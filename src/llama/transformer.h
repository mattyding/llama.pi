#ifndef __TRANSFORMER_H__
#define __TRANSFORMER_H__
// C struct definitions for the transformer model
#include "rpi.h"
#include "quant.h"

typedef struct {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
} Config;

typedef struct {
    // current wave of activations
    float *x; // activation at current time stamp (dim,)
    float *xb; // same, but inside a residual branch (dim,)
    float *xb2; // an additional buffer just for convenience (dim,)
    float *hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *q; // query (dim,)
    float *k; // key (dim,)
    float *v; // value (dim,)
    float *att; // buffer for scores/attention values (n_heads, seq_len)
    float *logits; // output logits
} RunState;

typedef struct {
    // current wave of activations
    float *x; // activation at current time stamp (dim,)
    float *xb; // same, but inside a residual branch (dim,)
    float *xb2; // an additional buffer just for convenience (dim,)
    float *hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    QuantizedTensor xq; // quantized x (dim,)
    QuantizedTensor hq; // quantized hb (hidden_dim,)
    float *q; // query (dim,)
    float *k; // key (dim,)
    float *v; // value (dim,)
    float *att; // buffer for scores/attention values (n_heads, seq_len)
    float *logits; // output logits
} qRunState;

// layerweights
typedef struct {
    // holds the weights for an individual transformer layer
    float* wq; // (dim,  head_size)
    float* wk; // (dim,  head_size)
    float* wv; // (dim,  head_size)
    float* wo; // (head_size, dim)
    float* w1; // (dim, hidden_dim)
    float* w2; // (hidden_dim, dim)
    float* w3; // (dim, hidden_dim)
    float* attention_norm_weight; // (dim,)
    float* ffn_norm_weight; // (dim,)
} LayerWeights;

typedef struct {
    // holds the quantized weights for an individual transformer layer
    QuantizedTensor wq; // (dim,  head_size)
    QuantizedTensor wk; // (dim,  head_size)
    QuantizedTensor wv; // (dim,  head_size)
    QuantizedTensor wo; // (head_size, dim)
    QuantizedTensor w1; // (dim, hidden_dim)
    QuantizedTensor w2; // (hidden_dim, dim)
    QuantizedTensor w3; // (dim, hidden_dim)
    float* attention_norm_weight; // (dim,)
    float* ffn_norm_weight; // (dim,)
} qLayerWeights;

#endif