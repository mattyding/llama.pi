#include <ctype.h>
#include <math.h>
#include <string.h>

#include "fileutils.h"
#include "prof.h"
#include "transformer.h"
#include "sampler.c"
#include "tokenizer.c"
#include "forward.c"
#include "forwardq.c"
#include "prune.h"

char *out_file_path = "../logs/gen_text.txt";

// experiment parameters
char *prompt = "hello world";
int use_forwardSeg = 0;     // 1 if use forwardSeg or 0 if use forward
int use_quantize = 1;       // 1 if use int8 quantization or 0 if use fp32
extern int use_prune;       // 1 if using pruned weights, modify in prune.h
int prune_assert = 0;       // use this to check if pruning is set correctly

float temperature = 1.0f;   // 0.0 = greedy deterministic. 1.0 = original. don't set higher
float topp = 0.9f;          // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
int steps = 256;            // number of steps to run for
unsigned long long rng_seed;

void generate(Config *config, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps) {
    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }
    // load out file in append mode
    FILE *out_file = fopen(out_file_path, "a");

    TransformerWeights fw;
    LayerWeights lw;
    RunState s;

    qTransformerWeights qw;
    qLayerWeights qlw;
    qRunState qs;
    
    if (!use_quantize) {
        if (!use_forwardSeg) {
            printf("Loading full weights\n");
            load_full_weights_fp32(config, &fw);
        } else {
            // load weights for an individual transformer layer
            // does not allocate additional memory; assumes malloc already done
            printf("Loading weights in segments\n");
            malloc_layer_weights(config, &lw);
        }
        stop_timer();
        print_time_elapsed("Weights load:", 0);

        malloc_run_state(config, &s);
    } else {
        if (!use_forwardSeg) {
            load_full_weights_q80(config, &qw);
            malloc_qrun_state(config, &qs);
        } else {
            malloc_qlayer_weights(config, &qlw);
            // these two sets of weights are not layer-specific so we load them at the beginning
            load_token_embedding_table(config, &qlw);
            load_rms_final_weight(config, &qlw);
            load_wcls(config, &qlw);
            malloc_qrun_state(config, &qs);
        }
    }

    print_mem_prof("Memory allocated for weights and run states before forward:");

    // start the main loop
    long start = 0;  // used to time our code, only initialized after first iteration
    int next;        // will store the next token in the sequence
    int token = prompt_tokens[0]; // kick off with the first token in the prompt
    int pos = 0;     // position in the sequence
    float *logits;
    while (pos < steps) {
        // forward the transformer to get logits for the next token
        if (!use_quantize) {
            if (!use_forwardSeg) {
                // standard fp32 forward pass
                logits = forward(config, &fw, &s, token, pos);
            } else {
                // warning: poss buggy
                // logits = forwardSeg(config, &lw, &s, token, pos);
                fprintf(stderr, "forwardSeg buggy. don't use at this time.\n");
                exit(EXIT_FAILURE);
            }
        } else {
            if (!use_forwardSeg) {
                // standard int8 quantized forward pass
                logits = forwardq(config, &qw, &qs, token, pos);
            } else {
                logits = forwardSegq(config, &qlw, &qs, token, pos);
            }
        }
        // advance the state machine
        if (pos < num_prompt_tokens - 1) {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos + 1];
        } else {
            // otherwise sample the next token from the logits
            next = sample(sampler, logits);
        }
        pos++;

        // data-dependent terminating condition: the BOS (=1) token delimits sequences
        if (next == 1) { break; }

        // print the token as string, decode it with the Tokenizer object
        char* piece = decode(tokenizer, token, next);
        safe_fprintf(out_file, piece); // same as fprintf(file, "%s", piece), but skips "unsafe" bytes
        fflush(stdout);
        token = next;

        // init the timer here because the first iteration can be slower
        if (start == 0) { start_timer(); start = -1;}
        printf("generated token %d\n", pos);
    }
    printf("\n");

    // report achieved tok/s (pos-1 because the timer starts after first iteration)
    if (pos > 1) {
        stop_timer();
        printf("Generated %d tokens in ", pos - 1);
        print_time_elapsed(NULL, pos - 1);
        print_mem_prof("Total mem usage on forward:");
    }

    fprintf(out_file, "\n");
    fclose(out_file);

    // free() fns mildly buggy so program terminates without freeing weights
    // we hope OS cleans up after us :) 
    // v bad practice >:( but alas i am on a deadline
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <use_forwardSeg> <use_quantize>\n", argv[0]);
        return 1;
    }
    use_forwardSeg = atoi(argv[1]);
    use_quantize = atoi(argv[2]);

    if (use_prune != prune_assert) {
        fprintf(stderr, "prune assert failed. check it is set correctly in prune.h.\n");
        exit(EXIT_FAILURE);
    }
    rng_seed = 0; //__rdtsc();
    printf("use_forwardSeg: %d\n", use_forwardSeg);
    printf("use_quantize: %d\n", use_quantize);
    printf("use_prune: %d\n", use_prune);
    printf("sampling rng seed: %llu\n", rng_seed);

    Config config;
    start_timer();
    if (!use_quantize) {
        read_config(config_fp32_path, & config, 0x616B3332, 1);
    } else {
        read_config(config_q80_path, &config, 0x616B3830, 2);
    }
    stop_timer();
    print_time_elapsed("Config load:", 0);
    // sanity check config (we know values for llama)
    if (config.dim != 4096 || config.n_heads != 32 || config.n_layers != 32 || config.vocab_size < 0) {
        fprintf(stderr, "Sanity check failed for Llama config\n");
        exit(EXIT_FAILURE);
    }

    // build the Tokenizer via the tokenizer .bin file
    Tokenizer tokenizer;
    start_timer();
    build_tokenizer(&tokenizer, tokenizer_path, config.vocab_size);
    stop_timer();
    print_time_elapsed("Tokenizer init:", 0);
    print_mem_prof(NULL);
    reset_mem_prof();

    // build the Sampler
    Sampler sampler;
    start_timer();
    build_sampler(&sampler, config.vocab_size, temperature, topp, rng_seed);
    stop_timer();
    print_time_elapsed("Sampler init:", 0);
    print_mem_prof(NULL);
    reset_mem_prof();

    // run!
    generate(&config, &tokenizer, &sampler, prompt, steps);

    // memory and file handles cleanup
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    return 0;
}