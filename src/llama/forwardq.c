// quantized and segmented forward pass of transformer model
#include "rpi.h"
#include "transformer.h"
#include "util.h"
#include "quant.h"
#include "algo.h"

#include "pi-files.h"
#include "fat32.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define HEAD_IDX 0  // the single head we use for attention


void malloc_layer_weights(Config *p, qLayerWeights *weights) {
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int head_size = dim / p->n_heads;
    int hidden_dim = p->hidden_dim;
    // rms norms
    weights->attention_norm_weight = kmalloc(dim * sizeof(float));
    weights->ffn_norm_weight = kmalloc(dim * sizeof(float));
    // attn weights
    weights->wq.q = kmalloc(head_size * sizeof(int8_t));
    weights->wq.s = kmalloc(sizeof(float));
    weights->wk.q = kmalloc(kv_dim * head_size * sizeof(int8_t));
    weights->wk.s = kmalloc(kv_dim * sizeof(float));
    weights->wv.q = kmalloc(kv_dim * head_size * sizeof(int8_t));
    weights->wv.s = kmalloc(kv_dim * sizeof(float));
    weights->wo.q = kmalloc(head_size * dim * sizeof(int8_t));
    weights->wo.s = kmalloc(head_size * dim * sizeof(float));
    // ffn weights
    weights->w1.q = kmalloc(dim * hidden_dim * sizeof(int8_t));
    weights->w1.s = kmalloc(dim * hidden_dim * sizeof(float));
    weights->w2.q = kmalloc(hidden_dim * dim * sizeof(int8_t));
    weights->w2.s = kmalloc(hidden_dim * dim * sizeof(float));
    weights->w3.q = kmalloc(dim * hidden_dim * sizeof(int8_t));
    weights->w3.s = kmalloc(dim * hidden_dim * sizeof(float));
    // ensure all mallocs went fine
    if (!weights->attention_norm_weight || !weights->ffn_norm_weight
     || !weights->wq.q || !weights->wq.s || !weights->wk.q || !weights->wk.s
     || !weights->wv.q || !weights->wv.s || !weights->wo.q || !weights->wo.s
     || !weights->w1.q || !weights->w1.s || !weights->w2.q || !weights->w2.s
     || !weights->w3.q || !weights->w3.s) {
        panic("issue with initializing layer weights\n");
    }
}

void malloc_run_state(Config *p, qRunState *s) {
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    s->x = kmalloc(p->dim * sizeof(float));
    s->xb = kmalloc(p->dim * sizeof(float));
    s->xb2 = kmalloc(p->dim * sizeof(float));
    s->hb = kmalloc(p->hidden_dim * sizeof(float));
    s->hb2 = kmalloc(p->hidden_dim * sizeof(float));
    s->xq = (QuantizedTensor) { .q = kmalloc(p->dim * sizeof(int8_t)), .s = kmalloc(p->dim * sizeof(float)) };
    s->hq = (QuantizedTensor) { .q = kmalloc(p->hidden_dim * sizeof(int8_t)), .s = kmalloc(p->hidden_dim * sizeof(float)) };
    s->q = kmalloc(p->dim * sizeof(float));
    s->k = kmalloc(kv_dim * sizeof(float));
    s->v = kmalloc(kv_dim * sizeof(float));
    s->att = kmalloc(p->n_heads * p->seq_len * sizeof(float));
    s->logits = kmalloc(p->vocab_size * sizeof(float));
    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
     || !s->k || !s->v || !s->att || !s->logits) {
        panic("issue with initializing runstate\n");
    }
}

// given ptr to data location serialized qtensor, loads into q and s and increments data_ptr
void load_quantized_tensor(char **data_ptr, int8_t *q, float *s, int size) {
    // assumes next (size) bytes are uint8_t q values
    memcpy(q, *data_ptr, size * sizeof(int8_t));
    *data_ptr += size * sizeof(int8_t);
    // assumes next (size) bytes are float s values
    memcpy(s, *data_ptr, size * sizeof(float));
    *data_ptr += size * sizeof(float);
}

// load weights for an individual transformer layer
// does not allocate additional memory; assumes malloc already done
void load_layer_weights(Config *p, fat32_fs_t *fs, pi_dirent_t *weight_dir, int layer_idx, qLayerWeights *weights, qRunState *state) {
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int head_size = dim / p->n_heads;
    int hidden_dim = p->hidden_dim;

    char *layer_ptr = read_file(fs, weight_dir, "layer%d.bin");
    // first byte is the magic number (0xak80)
    assert(*(int*)layer_ptr == 0x616B3830);
    layer_ptr += 1;
    // second byte is layer id
    assert(*(int*)layer_ptr == layer_idx);
    layer_ptr += 1;
    // memcpy weights into their assorted buffers. first the rmsnorm weights
    memcpy(weights->attention_norm_weight, layer_ptr, p->dim * sizeof(float));
    layer_ptr += p->dim * sizeof(float);
    memcpy(weights->ffn_norm_weight, layer_ptr, p->dim * sizeof(float));
    layer_ptr += p->dim * sizeof(float);
    // now the quantized weights. first set are attn, then ffn
    load_quantized_tensor(&layer_ptr, weights->wq.q, weights->wq.s, head_size);
    load_quantized_tensor(&layer_ptr, weights->wk.q + (HEAD_IDX / p->n_kv_heads) * head_size, weights->wk.s + (HEAD_IDX / p->n_kv_heads), head_size);
    load_quantized_tensor(&layer_ptr, weights->wv.q + (HEAD_IDX / p->n_kv_heads) * head_size, weights->wv.s + (HEAD_IDX / p->n_kv_heads), head_size);
    load_quantized_tensor(&layer_ptr, weights->wo.q + HEAD_IDX * head_size, weights->wo.s + HEAD_IDX * head_size, head_size * dim);

    load_quantized_tensor(&layer_ptr, weights->w1.q, weights->w1.s, dim * hidden_dim);
    load_quantized_tensor(&layer_ptr, weights->w2.q, weights->w2.s, hidden_dim * dim);
    load_quantized_tensor(&layer_ptr, weights->w3.q, weights->w3.s, dim * hidden_dim);
}

float* forwardq(Config *p, int token, int pos) {
    qRunState* s = NULL;        // compiler doesn't complain
    printk("mallocing run state\n");
    malloc_run_state(p, s);     // set s
    qLayerWeights* w = NULL;    // compiler doesn't complain
    printk("mallocing layer weights\n");
    malloc_layer_weights(p, w); // set w

    printk("beginning forward pass\n");

    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int hidden_dim =  p->hidden_dim;
    int head_size = dim / p->n_heads;

    // init filesystem
    fat32_fs_t fs;
    pi_dirent_t root;
    pi_dirent_t *w_dir = init_fs_cd_model(&fs, &root);

    // load token embedding table
    float *token_embedding_table = kmalloc(p->vocab_size * p->dim * sizeof(float));
    char *emb_ptr = read_file(&fs, w_dir, "token_embedding_table.bin");
    // first byte is the magic number (0xak80)
    assert(*(int*)emb_ptr == 0x616B3830);
    emb_ptr += 4;
    memcpy(token_embedding_table, emb_ptr, p->vocab_size * p->dim * sizeof(float));
    // cpy token embedding into x
    memcpy(x, token_embedding_table + token*dim, dim * sizeof(float));

    // forward all the layers
    for(int l = 0; l < p->n_layers; l++) {
        printk("loading layer %d\n", l);
        load_layer_weights(p, &fs, w_dir, l, w, s);
        printk("finished with load of layer %d\n", l);

        // attention rmsnorm
        rmsnorm(s->xb, x, w->attention_norm_weight, dim);

        // qkv matmulqs for this position
        quantize(&s->xq, s->xb, dim);
        matmulq(s->q, &s->xq, &w->wq, dim, dim);
        matmulq(s->k, &s->xq, &w->wk, dim, kv_dim);
        matmulq(s->v, &s->xq, &w->wv, dim, kv_dim);

        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        for (int i = 0; i < dim; i+=2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
            for (int v = 0; v < rotn; v++) {
                float* vec = v == 0 ? s->q : s->k; // the vector to rotate (query or key)
                float v0 = vec[i];
                float v1 = vec[i+1];
                vec[i]   = v0 * fcr - v1 * fci;
                vec[i+1] = v0 * fci + v1 * fcr;
            }
        }

        // single-head attention
        // get the query vector
        float* q = s->q;
        // attention scores
        float* att = s->att;

        // iterate over all timesteps, including the current one
        for (int t = 0; t <= pos; t++) {
            // get the key vector at this timestep
            float* k = s->k + t * kv_dim;
            // calculate the attention score as the dot product of q and k
            float score = 0.0f;
            for (int i = 0; i < kv_dim; i++) {
                score += q[i] * k[i];
            }
            score /= sqrtf(kv_dim);
            // save the score to the attention buffer
            att[t] = score;
        }

        // softmax the scores to get attention weights, from 0..pos inclusively
        softmax(att, pos + 1);

        // weighted sum of the values, store back into xb
        float* xb = s->xb;
        memset(xb, 0, dim * sizeof(float));
        for (int t = 0; t <= pos; t++) {
            // get the value vector at this timestep
            float* v = s->v + t * kv_dim;
            // get the attention weight for this timestep
            float a = att[t];
            // accumulate the weighted value into xb
            for (int i = 0; i < kv_dim; i++) {
                xb[i] += a * v[i];
            }
        }

        // final matmul to get the output of the attention
        quantize(&s->xq, s->xb, dim);
        matmulq(s->xb2, &s->xq, &w->wo, dim, dim);

        // residual connection back into x
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }

        // ffn rmsnorm
        rmsnorm(s->xb, x, w->ffn_norm_weight, dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        quantize(&s->xq, s->xb, dim);
        matmulq(s->hb, &s->xq, &w->w1, dim, hidden_dim);
        matmulq(s->hb2, &s->xq, &w->w3, dim, hidden_dim);

        // SwiGLU non-linearity
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= (1.0f / (1.0f + expf(-val)));
            // elementwise multiply with w3(x)
            val *= s->hb2[i];
            s->hb[i] = val;
        }

        // final matmul to get the output of the ffn
        quantize(&s->hq, s->hb, hidden_dim);
        matmulq(s->xb, &s->hq, &w->w2, hidden_dim, dim);

        // residual connection
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
    }

    // load rms_final_weight from file to w->ffn_norm_weight
    // saves minimal memory by avoiding the malloc call
    char *rms_final_weight_ptr = read_file(&fs, w_dir, "norm.bin");
    // first byte is the magic number (0xak80)
    assert(*(int*)rms_final_weight_ptr == 0x616B3830);
    rms_final_weight_ptr += 4;
    memcpy(w->ffn_norm_weight, rms_final_weight_ptr, dim * sizeof(float));

    // final rmsnorm
    rmsnorm(s->xb, x, w->ffn_norm_weight, dim);

    // classifier into logits
    quantize(&s->xq, x, dim);
    matmulq(s->logits, &s->xq, &w->wo, dim, p->vocab_size);
    return s->logits;
}