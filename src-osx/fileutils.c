#include "fileutils.h"

void read_config(char* config_path, Config* config) {
    FILE *file = fopen(config_path, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", config_path); exit(EXIT_FAILURE); }
    int magic_number;
    if (fread(&magic_number, sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
    if (magic_number != 0x616b3332) { fprintf(stderr, "magic number mismatch\n"); exit(EXIT_FAILURE); }
    int version;
    if (fread(&version, sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
    if (version != 1) { fprintf(stderr, "version number mismatch\n"); exit(EXIT_FAILURE); }
    if (fread(config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }
}

void memory_map_weights(TransformerWeights *w, Config* p, float* ptr, int shared_weights) {
    int head_size = p->dim / p->n_heads;
    // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
    unsigned long long n_layers = p->n_layers;
    w->token_embedding_table = ptr;
    ptr += p->vocab_size * p->dim;
    w->rms_att_weight = ptr;
    ptr += n_layers * p->dim;
    w->wq = ptr;
    ptr += n_layers * p->dim * (p->n_heads * head_size);
    w->wk = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wv = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wo = ptr;
    ptr += n_layers * (p->n_heads * head_size) * p->dim;
    w->rms_ffn_weight = ptr;
    ptr += n_layers * p->dim;
    w->w1 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->w2 = ptr;
    ptr += n_layers * p->hidden_dim * p->dim;
    w->w3 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->rms_final_weight = ptr;
    ptr += p->dim;
    ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_real (for RoPE)
    ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_imag (for RoPE)
    w->wcls = shared_weights ? w->token_embedding_table : ptr;
}

void load_full_weights_fp32(Config *p, TransformerWeights *w) {
    FILE *file = fopen(full_weights_fp32_path, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", full_weights_fp32_path); exit(EXIT_FAILURE); }
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    float *ptr = (float *)malloc(file_size);
    if (!ptr) { fprintf(stderr, "Couldn't allocate memory for weights\n"); exit(EXIT_FAILURE); }
    if (fread(ptr, file_size, 1, file) != 1) { exit(EXIT_FAILURE); }
    fclose(file);
    memory_map_weights(w, p, ptr, 1);
}