#include <ctype.h>
#include <math.h>
#include <string.h>

#include "fileutils.h"
#include "prof.h"
#include "transformer.h"
#include "forward.c"
#include "sampler.c"
#include "tokenizer.c"


// experiment parameters
char *prompt = "hello world";
char *checkpoint_path = NULL;  // e.g. out/model.bin
int use_forwardSeg = 0;        // 1 if use forwardSeg or 0 if use forward

float temperature = 1.0f;   // 0.0 = greedy deterministic. 1.0 = original. don't set higher
float topp = 0.9f;          // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
int steps = 256;            // number of steps to run for
unsigned long long rng_seed = 0; // seed rng with time by default

void generate(Config *config, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps) {
    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    TransformerWeights fw;
    LayerWeights lw;

    if (!use_forwardSeg) {
        printf("Loading full weights\n");
        load_full_weights_fp32(config, &fw);
    } else {
        // load weights for an individual transformer layer
        // does not allocate additional memory; assumes malloc already done
        printf("Loading weights in segments\n");
        malloc_layer_weights(config, &lw);
    }
    RunState s;
    malloc_run_state(config, &s);

    // start the main loop
    long start = 0;  // used to time our code, only initialized after first iteration
    int next;        // will store the next token in the sequence
    int token = prompt_tokens[0]; // kick off with the first token in the prompt
    int pos = 0;     // position in the sequence
    float *logits;
    while (pos < steps) {
        // forward the transformer to get logits for the next token
        if (use_forwardSeg)
            logits = forwardSeg(config, &lw, &s, token, pos);
        else
            logits = forward(config, &fw, &s, token, pos);

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
        safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
        fflush(stdout);
        token = next;

        // init the timer here because the first iteration can be slower
        if (start == 0) { start_timer(); }
        printf("fwd %d: ", pos);
        print_mem_prof(NULL);
    }
    printf("\n");

    // report achieved tok/s (pos-1 because the timer starts after first iteration)
    if (pos > 1) {
        stop_timer();
        print_time_elapsed(NULL, pos - 1);
        print_mem_prof("Total mem usage on forward:");
    }

    free(prompt_tokens);
    free_run_state(&s);
}

int main(int argc, char *argv[]) {
    use_forwardSeg = atoi(argv[1]);

    Config config;
    start_timer();
    read_config(config_fp32_path, &config);
    stop_timer();
    print_time_elapsed("Config load:", 0);

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