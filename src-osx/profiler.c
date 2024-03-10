// memory profiler that estimates the amount of RAM to run a model by tracing malloc calls
// personal implementation: nothing fancy, just ctrl+f malloc and add up the bytes
#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>

// can also be passed as command line arguments
char *checkpoint_path = NULL;  // e.g. out/model.bin
char *tokenizer_path = "tokenizer.bin";

int QUANTIZED = 0;

// ------------------ structs ------------------
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
    int8_t* q;    // quantized values
    float* s; // scaling factors
} QuantizedTensor;

typedef struct {
    float prob;
    int index;
} ProbIndex; 

// ------------------ functions ------------------

void print_config(Config* config) {
    printf("Model config:\n");
    printf("  dim: %d\n", config->dim);
    printf("  hidden_dim: %d\n", config->hidden_dim);
    printf("  n_heads: %d\n", config->n_heads);
    printf("  n_layers: %d\n", config->n_layers);
    printf("  seq_len: %d\n", config->seq_len);
    printf("  n_kv_heads: %d\n", config->n_kv_heads);
    printf("  vocab_size: %d\n", config->vocab_size);
}

unsigned long profile_run_state_q8(char* checkpoint, Config* config) {
    FILE *file = fopen(checkpoint, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint); exit(1); }
    // read in magic number (uint32), has to be 0x616b3432, i.e. "ak42" in ASCII
    uint32_t magic_number;
    if (fread(&magic_number, sizeof(uint32_t), 1, file) != 1) { exit(1); }
    if (magic_number != 0x616b3432) { fprintf(stderr, "Bad magic number, got %x instead.\n", magic_number); exit(1); }
    // read in the version number (uint32), has to be 2
    int version;
    if (fread(&version, sizeof(int), 1, file) != 1) { exit(1); }
    if (version != 2) { fprintf(stderr, "Bad version %d, need version 2\n", version); exit(1); }
    int header_size = 256; // the header size for version 2 in bytes
    // read in the Config
    if (fread(config, sizeof(Config), 1, file) != 1) { exit(1); }
    // read in flags
    uint8_t shared_classifier; // a byte to indicate if the classifier is shared
    if (fread(&shared_classifier, sizeof(uint8_t), 1, file) != 1) { exit(1); }
    int group_size; // the group size used in quantization
    if (fread(&group_size, sizeof(int), 1, file) != 1) { exit(1); }
    // figure out the file size
    fseek(file, 0, SEEK_END); // move file pointer to end of file
    ssize_t file_size = ftell(file); // get the file size, in bytes
    fclose(file);
    
    // print statistics
    print_config(config);
    printf("  shared_classifier: %d\n", shared_classifier);
    printf("  file_size: %ld bytes\n", file_size);


    unsigned long run_state_bytes = 0;
    int kv_dim = (config->dim * config->n_kv_heads) / config->n_heads;
    run_state_bytes += config->dim * sizeof(float); // s->x
    run_state_bytes += config->dim * sizeof(float); // s->xb
    run_state_bytes += config->dim * sizeof(float); // s->xb2
    run_state_bytes += config->hidden_dim * sizeof(float); // s->hb
    run_state_bytes += config->hidden_dim * sizeof(float); // s->hb2
    run_state_bytes += config->dim * sizeof(int8_t); // s->xq.q
    run_state_bytes += config->dim * sizeof(float); // s->xq.s
    run_state_bytes += config->hidden_dim * sizeof(int8_t); // s->hq.q
    run_state_bytes += config->hidden_dim * sizeof(float); // s->hq.s
    run_state_bytes += config->dim * sizeof(float); // s->q
    run_state_bytes += kv_dim * sizeof(float); // s->k
    run_state_bytes += kv_dim * sizeof(float); // s->v
    run_state_bytes += config->n_heads * config->seq_len * sizeof(float); // s->att
    run_state_bytes += config->vocab_size * sizeof(float); // s->logits
    run_state_bytes += config->n_layers * config->seq_len * kv_dim * sizeof(float); // s->key_cache
    run_state_bytes += config->n_layers * config->seq_len * kv_dim * sizeof(float); // s->value_cache
    printf("Run state size: %lu bytes\n", run_state_bytes);
    return run_state_bytes;
}

unsigned long profile_run_state_f16(char* checkpoint, Config* config) {
    FILE *file = fopen(checkpoint, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint); exit(EXIT_FAILURE); }
    // read in the config header
    if (fread(config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }
    // negative vocab size is hacky way of signaling unshared weights. bit yikes.
    int shared_weights = config->vocab_size > 0 ? 1 : 0;
    config->vocab_size = abs(config->vocab_size);
    // figure out the file size
    fseek(file, 0, SEEK_END); // move file pointer to end of file
    ssize_t file_size = ftell(file); // get the file size, in bytes
    fclose(file);

    // print statistics
    print_config(config);
    printf("  shared_weights: %d\n", shared_weights);
    printf("  file_size: %ld bytes\n", file_size);


    unsigned long run_state_bytes = 0;
    int kv_dim = (config->dim * config->n_kv_heads) / config->n_heads;
    run_state_bytes += config->dim * sizeof(float); // s->x
    run_state_bytes += config->dim * sizeof(float); // s->xb
    run_state_bytes += config->dim * sizeof(float); // s->xb2
    run_state_bytes += config->hidden_dim * sizeof(float); // s->hb
    run_state_bytes += config->hidden_dim * sizeof(float); // s->hb2
    run_state_bytes += config->dim * sizeof(float); // s->q
    run_state_bytes += config->n_heads * config->seq_len * sizeof(float); // s->att
    run_state_bytes += config->vocab_size * sizeof(float); // s->logits
    run_state_bytes += config->n_layers * config->seq_len * kv_dim * sizeof(float); // s->key_cache
    run_state_bytes += config->n_layers * config->seq_len * kv_dim * sizeof(float); // s->value_cache
    printf("Run state size: %lu bytes\n", run_state_bytes);
    return run_state_bytes;
}

unsigned long profile_weights_q8(Config* config) {
    // sum malloc calls from init_quantized_tensors
    unsigned long weights_bytes = 0;
    int head_size = config->dim / config->n_heads;
    weights_bytes += config->vocab_size * config->dim * sizeof(int8_t); // q_tokens.q
    weights_bytes += config->vocab_size * config->dim * sizeof(float); // q_tokens.s
    weights_bytes += config->vocab_size * config->dim * sizeof(float); // token_embedding_table
    weights_bytes += config->n_layers * config->dim * config->n_heads * head_size * sizeof(int8_t); // wq.q
    weights_bytes += config->n_layers * config->dim * config->n_heads * head_size * sizeof(float); // wq.s
    weights_bytes += config->n_layers * config->dim * config->n_kv_heads * head_size * sizeof(int8_t); // wk.q
    weights_bytes += config->n_layers * config->dim * config->n_kv_heads * head_size * sizeof(float); // wk.s
    weights_bytes += config->n_layers * config->dim * config->n_kv_heads * head_size * sizeof(int8_t); // wv.q
    weights_bytes += config->n_layers * config->dim * config->n_kv_heads * head_size * sizeof(float); // wv.s
    weights_bytes += config->n_layers * config->dim * config->n_heads * head_size * sizeof(int8_t); // wo.q
    weights_bytes += config->n_layers * config->dim * config->n_heads * head_size * sizeof(float); // wo.s
    weights_bytes += config->n_layers * config->dim * config->hidden_dim * sizeof(int8_t); // w1.q
    weights_bytes += config->n_layers * config->dim * config->hidden_dim * sizeof(float); // w1.s
    weights_bytes += config->n_layers * config->hidden_dim * config->dim * sizeof(int8_t); // w2.q
    weights_bytes += config->n_layers * config->hidden_dim * config->dim * sizeof(float); // w2.s
    weights_bytes += config->n_layers * config->hidden_dim * config->dim * sizeof(int8_t); // w3.q
    weights_bytes += config->n_layers * config->hidden_dim * config->dim * sizeof(float); // w3.s
    weights_bytes += config->dim * config->vocab_size * sizeof(int8_t); // wcls.q
    weights_bytes += config->dim * config->vocab_size * sizeof(float); // wcls.s
    printf("Weights size: %lu bytes\n", weights_bytes);
    return weights_bytes;
}

unsigned long profile_weights_f16(Config* config) {
    unsigned long weights_bytes = 0;
    int head_size = config->dim / config->n_heads;
    weights_bytes += config->vocab_size * config->dim * sizeof(float); // q_tokens
    weights_bytes += config->vocab_size * config->dim * sizeof(float); // token_embedding_table
    weights_bytes += config->n_layers * config->dim * config->n_heads * head_size * sizeof(float); // wq
    weights_bytes += config->n_layers * config->dim * config->n_kv_heads * head_size * sizeof(float); // wk
    weights_bytes += config->n_layers * config->dim * config->n_kv_heads * head_size * sizeof(float); // wv
    weights_bytes += config->n_layers * config->dim * config->n_heads * head_size * sizeof(float); // wo
    weights_bytes += config->n_layers * config->dim * config->hidden_dim * sizeof(float); // w1
    weights_bytes += config->n_layers * config->hidden_dim * config->dim * sizeof(float); // w2
    weights_bytes += config->n_layers * config->hidden_dim * config->dim * sizeof(float); // w3
    weights_bytes += config->dim * config->vocab_size * sizeof(float); // wcls
    printf("Weights size: %lu bytes\n", weights_bytes);
    return weights_bytes;
}

unsigned long profile_tokenizer(char * tokenizer_path, int vocab_size) {
    unsigned long tokenizer_bytes = 0;
    tokenizer_bytes += vocab_size * sizeof(char*); // t->vocab
    tokenizer_bytes += vocab_size * sizeof(float); // t->vocab_scores
    tokenizer_bytes += 256 * 2 * sizeof(char); // t->byte_pieces
    FILE *file = fopen(tokenizer_path, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", tokenizer_path); exit(1); }
    int max_token_length;
    if (fread(&max_token_length, sizeof(int), 1, file) != 1) { exit(1); }
    int len;
    for (int i = 0; i < vocab_size; i++) {
        tokenizer_bytes += max_token_length * sizeof(char); // t->vocab[i]
    }
    fclose(file);
    printf("Tokenizer size: %lu bytes\n", tokenizer_bytes);
    return tokenizer_bytes;
}

unsigned long profile_sampler(int vocab_size) {
    unsigned long sampler_bytes = 0;
    sampler_bytes += vocab_size * sizeof(ProbIndex); // probindex
    printf("Sampler size: %lu bytes\n", sampler_bytes);
    return sampler_bytes;
}

void error_usage() {
    printf("Usage: profiler <checkpoint_path> (optional)<tokenizer_path>\n");
    exit(1);
}

int main(int argc, char **argv) {
    // argparse
    if (argc >= 2) { checkpoint_path = argv[1]; } else if (checkpoint_path == NULL) { error_usage(); }
    if (argc >= 3) { tokenizer_path = argv[2]; }

    if (QUANTIZED) {
        Config config;
        unsigned long run_state_bytes = profile_run_state_q8(checkpoint_path, &config);
        unsigned long weights_bytes = profile_weights_q8(&config);
        unsigned long tokenizer_bytes = profile_tokenizer(tokenizer_path, config.vocab_size);
        unsigned long sampler_bytes = profile_sampler(config.vocab_size);
        unsigned long total_bytes = run_state_bytes + weights_bytes + tokenizer_bytes + sampler_bytes;
        printf("Minimum Total RAM needed:   %lu bytes (%luMB)\n", total_bytes, total_bytes / 1000000);
    } else {
        Config config;
        unsigned long run_state_bytes = profile_run_state_f16(checkpoint_path, &config);
        unsigned long weights_bytes = profile_weights_f16(&config);
        unsigned long tokenizer_bytes = profile_tokenizer(tokenizer_path, config.vocab_size);
        unsigned long sampler_bytes = profile_sampler(config.vocab_size);
        unsigned long total_bytes = run_state_bytes + weights_bytes + tokenizer_bytes + sampler_bytes;
        printf("Minimum Total RAM needed:   %lu bytes (%luMB)\n", total_bytes, total_bytes / 1000000);
    }
    return 0;
}