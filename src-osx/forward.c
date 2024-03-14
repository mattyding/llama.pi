#include <stdlib.h>
#include <stdio.h>
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
#include "transformer.h"
#include "prof.h"

static void malloc_run_state(Config* p, RunState* s) {
    // we calloc instead of malloc to keep valgrind happy
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    s->x = calloc(p->dim, sizeof(float));
    s->xb = calloc(p->dim, sizeof(float));
    s->xb2 = calloc(p->dim, sizeof(float));
    s->hb = calloc(p->hidden_dim, sizeof(float));
    s->hb2 = calloc(p->hidden_dim, sizeof(float));
    s->q = calloc(p->dim, sizeof(float));
    s->key_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->value_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->att = calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = calloc(p->vocab_size, sizeof(float));
    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
     || !s->key_cache || !s->value_cache || !s->att || !s->logits) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
}

static void malloc_layer_weights(Config *p, LayerWeights *w) {
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int n_kv_heads = p->n_kv_heads;
    int head_size = dim / p->n_heads;
    int hidden_dim = p->hidden_dim;
    w->rms_att_weight = calloc(dim, sizeof(float));
    w->rms_ffn_weight = calloc(dim, sizeof(float));
    w->wq = calloc(dim * n_kv_heads * head_size, sizeof(float));
    w->wk = calloc(dim * n_kv_heads * head_size, sizeof(float));
    w->wv = calloc(dim * n_kv_heads * head_size, sizeof(float));
    w->wo = calloc(dim * head_size, sizeof(float));
    w->w1 = calloc(dim * hidden_dim, sizeof(float));
    w->w2 = calloc(hidden_dim * dim, sizeof(float));
    w->w3 = calloc(dim * hidden_dim, sizeof(float));
}

static void free_run_state(RunState* s) {
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->q);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
}

static void free_layer_weights(LayerWeights *w) {
    free(w->rms_att_weight);
    free(w->rms_ffn_weight);
    free(w->wq);
    free(w->wk);
    free(w->wv);
    free(w->wo);
    free(w->w1);
    free(w->w2);
    free(w->w3);
}

static float* forward(Config *p, TransformerWeights *w, RunState *s, int token, int pos) {
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int hidden_dim =  p->hidden_dim;
    int head_size = dim / p->n_heads;

    // copy the token embedding into x
    float* content_row = w->token_embedding_table + token * dim;
    memcpy(x, content_row, dim*sizeof(*x));

    // forward all the layers
    for(unsigned long long l = 0; l < p->n_layers; l++) {

        stop_timer();
        print_time_elapsed("starting layer", 0);

        // attention rmsnorm
        rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);

        // key and value point to the kv cache
        int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
        s->k = s->key_cache + loff + pos * kv_dim;
        s->v = s->value_cache + loff + pos * kv_dim;

        // qkv matmuls for this position
        matmulf(s->q, s->xb, w->wq + l*dim*dim, dim, dim);
        matmulf(s->k, s->xb, w->wk + l*dim*kv_dim, dim, kv_dim);
        matmulf(s->v, s->xb, w->wv + l*dim*kv_dim, dim, kv_dim);

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
        matmulf(s->xb2, s->xb, w->wo + l*dim*dim, dim, dim);

        // residual connection back into x
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }

        // ffn rmsnorm
        rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        matmulf(s->hb, s->xb, w->w1 + l*dim*hidden_dim, dim, hidden_dim);
        matmulf(s->hb2, s->xb, w->w3 + l*dim*hidden_dim, dim, hidden_dim);

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
        matmulf(s->xb, s->hb, w->w2 + l*dim*hidden_dim, hidden_dim, dim);

        // residual connection
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
    }

    stop_timer();
    print_time_elapsed("finished all layers", 0);

    // final rmsnorm
    rmsnorm(x, x, w->rms_final_weight, dim);

    // classifier into logits
    matmulf(s->logits, x, w->wcls, p->dim, p->vocab_size);
    return s->logits;
}

static void load_layer_weights(Config *p, int layer_idx, LayerWeights *weights) {
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int n_kv_heads = p->n_kv_heads;
    int head_size = dim / p->n_heads;
    int hidden_dim = p->hidden_dim;
    int fd;
    char filename[64];
    sprintf(filename, "../models//7b-f32/layer%d.bin", layer_idx);
    fd = open(filename, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "failed to open %s\n", filename);
        exit(EXIT_FAILURE);
    }
    // first byte is magic number
    int magic;
    read(fd, &magic, sizeof(int));
    if (magic != 0x616B3332) {
        fprintf(stderr, "magic number mismatch in %s\n", filename);
        exit(EXIT_FAILURE);
    }
    // second byte is layer id
    int layer_id;
    read(fd, &layer_id, sizeof(int));
    if (layer_id != layer_idx) {
        fprintf(stderr, "layer id mismatch in %s\n", filename);
        exit(EXIT_FAILURE);
    }
    // rms_att_weight
    read(fd, weights->rms_att_weight, dim * sizeof(float));
    // rms_ffn_weight
    read(fd, weights->rms_ffn_weight, dim * sizeof(float));
    // wq
    read(fd, weights->wq, dim * n_kv_heads * head_size * sizeof(float));
    // wk
    read(fd, weights->wk, dim * n_kv_heads * head_size * sizeof(float));
    // wv
    read(fd, weights->wv, dim * n_kv_heads * head_size * sizeof(float));
    // wo
    read(fd, weights->wo, dim * head_size * sizeof(float));
    // w1
    read(fd, weights->w1, dim * hidden_dim * sizeof(float));
    // w2
    read(fd, weights->w2, hidden_dim * dim * sizeof(float));
    // w3
    read(fd, weights->w3, dim * hidden_dim * sizeof(float));
    close(fd);
}

// weights are iteratively loaded per layer. minimizes stack memory
static float* forwardSeg(Config *p, LayerWeights *w, RunState *s, int token, int pos) {
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int hidden_dim =  p->hidden_dim;
    int head_size = dim / p->n_heads;


    // copy the token embedding into x
    // token embedding table at tok_emb.bin
    float *token_embedding_table = malloc(p->vocab_size * p->dim * sizeof(float));
    int fd = open("../models/7b-f32/tok_emb.bin", O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "failed to open tok_emb.bin\n");
        exit(EXIT_FAILURE);
    }
    // first byte is magic number
    int magic;
    read(fd, &magic, sizeof(int));
    if (magic != 0x616B3332) {
        fprintf(stderr, "magic number mismatch in tok_emb.bin\n");
        exit(EXIT_FAILURE);
    }
    // copy the token embedding table into memory
    read(fd, token_embedding_table, p->vocab_size * p->dim * sizeof(float));
    close(fd);

    // forward all the layers
    for(unsigned long long l = 0; l < p->n_layers; l++) {

        printf("loading layer %llu\n", l);

        load_layer_weights(p, l, w);

        // attention rmsnorm
        rmsnorm(s->xb, x, w->rms_att_weight, dim);

        // key and value point to the kv cache
        int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
        s->k = s->key_cache + loff + pos * kv_dim;
        s->v = s->value_cache + loff + pos * kv_dim;

        // qkv matmuls for this position
        matmulf(s->q, s->xb, w->wq + l*dim*dim, dim, dim);
        matmulf(s->k, s->xb, w->wk + l*dim*kv_dim, dim, kv_dim);
        matmulf(s->v, s->xb, w->wv + l*dim*kv_dim, dim, kv_dim);


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
        matmulf(s->xb2, s->xb, w->wo + l*dim*dim, dim, dim);

        // residual connection back into x
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }

        // ffn rmsnorm
        rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        matmulf(s->hb, s->xb, w->w1 + l*dim*hidden_dim, dim, hidden_dim);
        matmulf(s->hb2, s->xb, w->w3 + l*dim*hidden_dim, dim, hidden_dim);

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
        matmulf(s->xb, s->hb, w->w2 + l*dim*hidden_dim, hidden_dim, dim);

        // residual connection
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
    }

    // final rms norm weights saved in norm.bin
    float *rms_final_weight = malloc(p->dim * sizeof(float));
    fd = open("../models/7b-f32/norm.bin", O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "failed to open norm.bin\n");
        exit(EXIT_FAILURE);
    }
    // first byte is magic number
    read(fd, &magic, sizeof(int));
    if (magic != 0x616B3830) {
        fprintf(stderr, "magic number mismatch in norm.bin\n");
        exit(EXIT_FAILURE);
    }
    // copy the final rms norm weights into memory
    read(fd, rms_final_weight, p->dim * sizeof(float));
    close(fd);
    
    // final rmsnorm
    rmsnorm(s->xb, x, rms_final_weight, dim);

    // output weights saved in output.bin
    float *wcls = malloc(p->dim * p->vocab_size * sizeof(float));
    fd = open("../models/7b-f32/output.bin", O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "failed to open output.bin\n");
        exit(EXIT_FAILURE);
    }
    // first byte is magic number
    read(fd, &magic, sizeof(int));
    if (magic != 0x616B3332) {
        fprintf(stderr, "magic number mismatch in output.bin\n");
        exit(EXIT_FAILURE);
    }
    // copy the output weights into memory
    read(fd, wcls, p->dim * p->vocab_size * sizeof(float));
    close(fd);

    // classifier into logits
    matmulf(s->logits, x, wcls, dim, p->vocab_size);
    
    free_layer_weights(w);

    return s->logits;
}