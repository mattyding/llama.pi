/* Inference for Llama-2 Transformer model in pure C, int8 quantized forward pass. */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#if defined _WIN32
    #include "win.h"
#else
    #include <unistd.h>
    #include <sys/mman.h>
#endif

#include <x86intrin.h>
#include <time.h>

#include "algo.h"
#include "quant.h"
#include "transformer.h"
#include "prof.h"


static void malloc_qrun_state(Config *p, qRunState* s) {
    // we calloc instead of malloc to keep valgrind happy
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    s->x = calloc(p->dim, sizeof(float));
    s->xb = calloc(p->dim, sizeof(float));
    s->xb2 = calloc(p->dim, sizeof(float));
    s->hb = calloc(p->hidden_dim, sizeof(float));
    s->hb2 = calloc(p->hidden_dim, sizeof(float));
    s->xq = (QuantizedTensor) { .q = calloc(p->dim, sizeof(int8_t)), .s = calloc(p->dim, sizeof(float)) };
    s->hq = (QuantizedTensor) { .q = calloc(p->hidden_dim, sizeof(int8_t)), .s = calloc(p->hidden_dim, sizeof(float)) };
    s->q = calloc(p->dim, sizeof(float));
    s->k = calloc(kv_dim, sizeof(float));
    s->v = calloc(kv_dim, sizeof(float));
    s->att = calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = calloc(p->vocab_size, sizeof(float));
    s->key_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->value_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
     || !s->k || !s->v || !s->att || !s->logits || !s->key_cache
     || !s->value_cache) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
}

static void free_qrun_state(qRunState* s) {
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->xq.q);
    free(s->xq.s);
    free(s->hq.q);
    free(s->hq.s);
    free(s->q);
    free(s->k);
    free(s->v);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
}

static void malloc_qlayer_weights(Config *p, qLayerWeights *w) {
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int n_kv_heads = p->n_kv_heads;
    int head_size = dim / p->n_heads;
    int hidden_dim = p->hidden_dim;

    // fp32 weights
    w->rms_att_weight = calloc(dim, sizeof(float));
    w->rms_ffn_weight = calloc(dim, sizeof(float));
    w->token_embedding_table = calloc(p->vocab_size * dim, sizeof(float));
    w->rms_final_weight = calloc(dim, sizeof(float));

    // quantized weights
    w->wq = calloc(dim * (p->n_heads * head_size), sizeof(QuantizedTensor));
    w->wk = calloc(dim * (n_kv_heads * head_size), sizeof(QuantizedTensor));
    w->wv = calloc(dim * (n_kv_heads * head_size), sizeof(QuantizedTensor));
    w->wo = calloc((p->n_heads * head_size) * dim, sizeof(QuantizedTensor));
    w->w1 = calloc(dim * hidden_dim, sizeof(QuantizedTensor));
    w->w2 = calloc(hidden_dim * dim, sizeof(QuantizedTensor));
    w->w3 = calloc(dim * hidden_dim, sizeof(QuantizedTensor));

    w->wcls = calloc(dim * p->vocab_size, sizeof(QuantizedTensor));
    // ensure all mallocs went fine
    if (!w->rms_att_weight || !w->rms_ffn_weight || !w->rms_final_weight
     || !w->wq || !w->wk || !w->wv || !w->wo || !w->w1 || !w->w2 || !w->w3
     || !w->wcls) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
}

static void free_qlayer_weights(qLayerWeights *w) {
    free(w->rms_att_weight);
    free(w->rms_ffn_weight);
    free(w->token_embedding_table);
    free(w->rms_final_weight);
    free(w->wq);
    free(w->wk);
    free(w->wv);
    free(w->wo);
    free(w->w1);
    free(w->w2);
    free(w->w3);
    free(w->wcls);
}

static void free_full_qweights(qTransformerWeights *w) {
    free(w->rms_att_weight);
    free(w->rms_ffn_weight);
    free(w->token_embedding_table);
    free(w->rms_final_weight);
    free(w->q_tokens);
    free(w->wq);
    free(w->wk);
    free(w->wv);
    free(w->wo);
    free(w->w1);
    free(w->w2);
    free(w->w3);
    free(w->wcls);
}

static float* forwardq(Config *p, qTransformerWeights *w, qRunState *s, int token, int pos) {
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int hidden_dim =  p->hidden_dim;
    int head_size = dim / p->n_heads;

    // copy the token embedding into x
    memcpy(x, w->token_embedding_table + token*dim, dim * sizeof(float));

    reset_timer();

    // forward all the layers
    for(int l = 0; l < p->n_layers; l++) {

        stop_timer();
        printf("starting layer %d:\n", l);
        print_time_elapsed(NULL, 0);

        // attention rmsnorm
        rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);

        // qkv matmuls for this position
        quantize(&s->xq, s->xb, dim);
        matmulq(s->q, &s->xq, w->wq + l, dim, dim);
        matmulq(s->k, &s->xq, w->wk + l, dim, kv_dim);
        matmulq(s->v, &s->xq, w->wv + l, dim, kv_dim);

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

        // save key,value at this time step (pos) to our kv cache
        int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
        float* key_cache_row = s->key_cache + loff + pos * kv_dim;
        float* value_cache_row = s->value_cache + loff + pos * kv_dim;
        memcpy(key_cache_row, s->k, kv_dim * sizeof(*key_cache_row));
        memcpy(value_cache_row, s->v, kv_dim * sizeof(*value_cache_row));

        // multihead attention. iterate over all heads
        int h;
        #pragma omp parallel for private(h)
        for (h = 0; h < p->n_heads; h++) {
            // get the query vector for this head
            float* q = s->q + h * head_size;
            // attention scores for this head
            float* att = s->att + h * p->seq_len;
            // iterate over all timesteps, including the current one
            for (int t = 0; t <= pos; t++) {
                // get the key vector for this head and at this timestep
                float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                // calculate the attention score as the dot product of q and k
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                score /= sqrtf(head_size);
                // save the score to the attention buffer
                att[t] = score;
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            softmax(att, pos + 1);

            // weighted sum of the values, store back into xb
            float* xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                // get the value vector for this head and at this timestep
                float* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                // get the attention weight for this timestep
                float a = att[t];
                // accumulate the weighted value into xb
                for (int i = 0; i < head_size; i++) {
                    xb[i] += a * v[i];
                }
            }
        }

        // final matmul to get the output of the attention
        quantize(&s->xq, s->xb, dim);
        matmulq(s->xb2, &s->xq, w->wo + l, dim, dim);

        // residual connection back into x
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }

        // ffn rmsnorm
        rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        quantize(&s->xq, s->xb, dim);
        matmulq(s->hb, &s->xq, w->w1 + l, dim, hidden_dim);
        matmulq(s->hb2, &s->xq, w->w3 + l, dim, hidden_dim);

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
        matmulq(s->xb, &s->hq, w->w2 + l, hidden_dim, dim);

        // residual connection
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
    }

    stop_timer();
    printf("finished all layers:\n");
    print_time_elapsed(NULL, 0);
    
    // final rmsnorm
    rmsnorm(x, x, w->rms_final_weight, dim);

    // classifier into logits
    quantize(&s->xq, x, dim);
    matmulq(s->logits, &s->xq, w->wcls, dim, p->vocab_size);
    return s->logits;
}

static void load_qlayer_weights(Config *p, int layer_idx, qLayerWeights *qlweights, void **map_start, int *file_size){
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int n_kv_heads = p->n_kv_heads;
    int head_size = dim / p->n_heads;
    int hidden_dim = p->hidden_dim;

    // layer file is at "../models/7b-q80/layer%d.bin"
    char *layer_path = "../models/7b-q80/layer";
    char layer_num[4];
    sprintf(layer_num, "%d", layer_idx);
    char *layer_ext = ".bin";
    char *layer_full_path = malloc(strlen(layer_path) + strlen(layer_num) + strlen(layer_ext) + 1);
    strcpy(layer_full_path, layer_path);
    strcat(layer_full_path, layer_num);
    strcat(layer_full_path, layer_ext);

    // get file size
    FILE *f = fopen(layer_full_path, "rb");
    if (!f) {
        fprintf(stderr, "Couldn't open file %s\n", layer_full_path);
        exit(EXIT_FAILURE);
    }
    fseek(f, 0, SEEK_END);
    *file_size = ftell(f);
    fclose(f);

    int file = open(layer_full_path, O_RDONLY);
    if (file == -1) {
        fprintf(stderr, "Couldn't open file %s\n", layer_full_path);
        exit(EXIT_FAILURE);
    }
    free(layer_full_path);
    void *ptr = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, file, 0);
    if (ptr == MAP_FAILED) {
        fprintf(stderr, "Couldn't map file %s\n", layer_full_path);
        exit(EXIT_FAILURE);
    }
    close(file);
    *map_start = ptr;

    // first 4 bytes is magic number
    int magic = *(int *)ptr;
    if (magic != 0x616B3830) {
        fprintf(stderr, "magic number mismatch\n");
        exit(EXIT_FAILURE);
    }
    ptr += sizeof(int);
    // second 4 bytes is layer idx
    int layer = *(int *)ptr;
    if (layer != layer_idx) {
        fprintf(stderr, "layer number mismatch\n");
        exit(EXIT_FAILURE);
    }
    ptr += sizeof(int);

    // now copy weights
    memcpy(qlweights->rms_att_weight, ptr, dim * sizeof(float));
    ptr += dim * sizeof(float);
    memcpy(qlweights->rms_ffn_weight, ptr, dim * sizeof(float));
    ptr += dim * sizeof(float);

    void *data = &ptr;
    load_quantized_tensors(data, qlweights->wq, 1, dim * (p->n_heads * head_size));
    load_quantized_tensors(data, qlweights->wk, 1, dim * (n_kv_heads * head_size));
    load_quantized_tensors(data, qlweights->wv, 1, dim * (n_kv_heads * head_size));
    load_quantized_tensors(data, qlweights->wo, 1, (p->n_heads * head_size) * dim);
    load_quantized_tensors(data, qlweights->w1, 1, dim * hidden_dim);
    load_quantized_tensors(data, qlweights->w2, 1, hidden_dim * dim);
    load_quantized_tensors(data, qlweights->w3, 1, dim * hidden_dim);
}


// weights are iteratively loaded per layer. minimizes stack memory
static float* forwardSegq(Config *p, qLayerWeights *qlw, qRunState *s, int token, int pos) {
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int hidden_dim =  p->hidden_dim;
    int head_size = dim / p->n_heads;

    // copy the token embedding into x
    memcpy(x, qlw->token_embedding_table + token*dim, dim * sizeof(float));

    // needed to free layer weight mem
    void *map_start;
    int file_size;

    // forward all the layers
    // qLayerWeights starts off uninitialized
    for(int l = 0; l < p->n_layers; l++) {
        stop_timer();
        printf("layer %d:\n", l);
        print_time_elapsed(NULL, 0);

        load_qlayer_weights(p, l, qlw, &map_start, &file_size);

        // attention rmsnorm
        rmsnorm(s->xb, x, qlw->rms_att_weight, dim);

        // qkv matmuls for this position
        quantize(&s->xq, s->xb, dim);
        matmulq(s->q, &s->xq, qlw->wq, dim, dim);
        matmulq(s->k, &s->xq, qlw->wk, dim, kv_dim);
        matmulq(s->v, &s->xq, qlw->wv, dim, kv_dim);

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

        // save key,value at this time step (pos) to our kv cache
        int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
        float* key_cache_row = s->key_cache + loff + pos * kv_dim;
        float* value_cache_row = s->value_cache + loff + pos * kv_dim;
        memcpy(key_cache_row, s->k, kv_dim * sizeof(*key_cache_row));
        memcpy(value_cache_row, s->v, kv_dim * sizeof(*value_cache_row));

        // multihead attention. iterate over all heads
        int h;
        #pragma omp parallel for private(h)
        for (h = 0; h < p->n_heads; h++) {
            // get the query vector for this head
            float* q = s->q + h * head_size;
            // attention scores for this head
            float* att = s->att + h * p->seq_len;
            // iterate over all timesteps, including the current one
            for (int t = 0; t <= pos; t++) {
                // get the key vector for this head and at this timestep
                float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                // calculate the attention score as the dot product of q and k
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                score /= sqrtf(head_size);
                // save the score to the attention buffer
                att[t] = score;
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            softmax(att, pos + 1);

            // weighted sum of the values, store back into xb
            float* xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                // get the value vector for this head and at this timestep
                float* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                // get the attention weight for this timestep
                float a = att[t];
                // accumulate the weighted value into xb
                for (int i = 0; i < head_size; i++) {
                    xb[i] += a * v[i];
                }
            }

        }

        // final matmul to get the output of the attention
        quantize(&s->xq, s->xb, dim);
        matmulq(s->xb2, &s->xq, qlw->wo, dim, dim);

        // residual connection back into x
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }

        // ffn rmsnorm
        rmsnorm(s->xb, x, qlw->rms_ffn_weight, dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        quantize(&s->xq, s->xb, dim);
        matmulq(s->hb, &s->xq, qlw->w1, dim, hidden_dim);
        matmulq(s->hb2, &s->xq, qlw->w3, dim, hidden_dim);


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
        matmulq(s->xb, &s->hq, qlw->w2, hidden_dim, dim);

        // residual connection
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }

        // free layer memory
        munmap(map_start, file_size);
    }

    stop_timer();
    print_time_elapsed("finished all layers", 0);

    // final rmsnorm
    rmsnorm(x, x, qlw->rms_final_weight, dim);

    // classifier into logits
    quantize(&s->xq, x, dim);
    matmulq(s->logits, &s->xq, qlw->wcls, dim, p->vocab_size);

    return s->logits;
}