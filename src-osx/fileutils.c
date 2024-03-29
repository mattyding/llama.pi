#include "fileutils.h"

void read_config(char* config_path, Config* config, int magic, int version) {
    FILE *file = fopen(config_path, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", config_path); exit(EXIT_FAILURE); }
    int x;
    if (fread(&x, sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
    if (x != magic) { fprintf(stderr, "magic number mismatch\n"); exit(EXIT_FAILURE); }
    if (fread(&x, sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
    if (x != version) { fprintf(stderr, "version number mismatch\n"); exit(EXIT_FAILURE); }

    if (fread(&config->dim, sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
    printf("config->dim: %d\n", config->dim);
    if (fread(&config->hidden_dim, sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
    printf("config->hidden_dim: %d\n", config->hidden_dim);
    if (fread(&config->n_layers, sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
    printf("config->n_layers: %d\n", config->n_layers);
    if (fread(&config->n_heads, sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
    printf("config->n_heads: %d\n", config->n_heads);
    if (fread(&config->n_kv_heads, sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
    printf("config->n_kv_heads: %d\n", config->n_kv_heads);
    if (fread(&config->vocab_size, sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
    printf("config->vocab_size: %d\n", config->vocab_size);
    if (fread(&config->seq_len, sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
    printf("config->seq_len: %d\n", config->seq_len);
    fclose(file);
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
    // ptr += p->dim;
    // ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_real (for RoPE)
    // ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_imag (for RoPE)
    w->wcls = shared_weights ? w->token_embedding_table : ptr;

    if (use_prune) {
        printf("pruning fp32 weights with ratio %f\n", prune_ratio);
        prune_matrix(w->w1, n_layers * p->dim * p->hidden_dim, prune_ratio);
        prune_matrix(w->w2, n_layers * p->hidden_dim * p->dim, prune_ratio);
        prune_matrix(w->w3, n_layers * p->dim * p->hidden_dim, prune_ratio);
    }
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
    memory_map_weights(w, p, ptr, shared_classifier);
}

void memory_map_qweights(qTransformerWeights *w, Config* p, void* ptr, uint8_t shared_classifier) {
    int head_size = p->dim / p->n_heads;
    // first are the parameters that are kept in fp32 (the rmsnorm (1D) weights)
    float* fptr = (float*) ptr; // cast our pointer to float*
    w->rms_att_weight = fptr;
    fptr += p->n_layers * p->dim;
    w->rms_ffn_weight = fptr;
    fptr += p->n_layers * p->dim;
    w->rms_final_weight = fptr;
    fptr += p->dim;

    // now read all the quantized weights
    ptr = (void*)fptr; // now cast the pointer back to void*
    w->q_tokens = init_quantized_tensors(&ptr, 1, p->vocab_size * p->dim);
    // dequantize token embedding table
    w->token_embedding_table = malloc(p->vocab_size * p->dim * sizeof(float));
    dequantize(w->q_tokens, w->token_embedding_table, p->vocab_size * p->dim);

    w->wq = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_heads * head_size));
    w->wk = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_kv_heads * head_size));
    w->wv = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_kv_heads * head_size));
    w->wo = init_quantized_tensors(&ptr, p->n_layers, (p->n_heads * head_size) * p->dim);

    w->w1 = init_quantized_tensors(&ptr, p->n_layers, p->dim * p->hidden_dim);
    w->w2 = init_quantized_tensors(&ptr, p->n_layers, p->hidden_dim * p->dim);
    w->w3 = init_quantized_tensors(&ptr, p->n_layers, p->dim * p->hidden_dim);

    w->wcls = shared_classifier ? w->q_tokens : init_quantized_tensors(&ptr, 1, p->dim * p->vocab_size);

    if (use_prune) {
        printf("pruning q80 weights with ratio %f\n", prune_ratio);
        prune_qmatrix(w->w1, p->n_layers, p->dim * p->hidden_dim, prune_ratio);
        prune_qmatrix(w->w2, p->n_layers, p->hidden_dim * p->dim, prune_ratio);
        prune_qmatrix(w->w3, p->n_layers, p->dim * p->hidden_dim, prune_ratio);
    }
}

void load_full_weights_q80(Config *p, qTransformerWeights *w) {
    FILE *file = fopen(full_weights_q80_path, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", full_weights_q80_path); exit(EXIT_FAILURE); }
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fclose(file);
    int fd = open(full_weights_q80_path, O_RDONLY);
    if (fd == -1) { fprintf(stderr, "Couldn't open file %s\n", full_weights_q80_path); exit(EXIT_FAILURE); }
    void *ptr = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (ptr == MAP_FAILED) { fprintf(stderr, "Couldn't map file %s\n", full_weights_q80_path); exit(EXIT_FAILURE); }
    memory_map_qweights(w, p, ptr, shared_classifier);
}

// stored in tok_emb.bin; not quantized
// first byte is magic
// then (vocab_size * dim) floats for the token embedding table
void load_token_embedding_table(Config *p, qLayerWeights *w) {
    FILE *file = fopen(tok_embed_q80_path, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", tok_embed_q80_path); exit(EXIT_FAILURE); }
    // first byte is magic
    int x;
    if (fread(&x, sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
    if (x != 0x616B3830) { fprintf(stderr, "magic number mismatch\n"); exit(EXIT_FAILURE); }

    // (vocab_size * dim) floats for the token embedding table
    // assume already allocated memory
    if (fread(w->token_embedding_table, sizeof(float), p->vocab_size * p->dim, file) != p->vocab_size * p->dim) { exit(EXIT_FAILURE); }
    fclose(file);
}

void load_rms_final_weight(Config *p, qLayerWeights *w) {
    FILE *file = fopen(rms_final_q80_path, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", rms_final_q80_path); exit(EXIT_FAILURE); }
    // first byte is magic
    int x;
    if (fread(&x, sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
    if (x != 0x616B3830) { fprintf(stderr, "magic number mismatch\n"); exit(EXIT_FAILURE); }

    // (dim * sizeof(float)) bytes for the final rmsnorm weights
    // assume already allocated memory
    if (fread(w->rms_final_weight, sizeof(float), p->dim, file) != p->dim) { exit(EXIT_FAILURE); }
    fclose(file);
}

// wcls stored at output.bin
void load_wcls(Config *p, qLayerWeights *w) {
    // open() and then memmap
    int file = open(wcls_q80_path, O_RDONLY);
    if (file == -1) { fprintf(stderr, "Couldn't open file %s\n", wcls_q80_path); exit(EXIT_FAILURE); }
    // memmap
    void *ptr = mmap(NULL, sizeof(int) + p->dim * p->vocab_size * sizeof(QuantizedTensor), PROT_READ, MAP_PRIVATE, file, 0);
    if (ptr == MAP_FAILED) { fprintf(stderr, "Couldn't map file %s\n", wcls_q80_path); exit(EXIT_FAILURE); }
    close(file);
    // first byte is magic
    int x = *(int*)ptr;
    if (x != 0x616B3830) { fprintf(stderr, "magic number mismatch\n"); exit(EXIT_FAILURE); }
    ptr += sizeof(int);
    // increment pointer
    // load quantized weights
    load_quantized_tensors(&ptr, w->wcls, 1, p->dim * p->vocab_size);
}