/* segmented inference for llama2
our r/pi doesn't have enough RAM to run the full model so we split inference into segments,
load weights and data one at a time, and run a forward pass on each segment

our r/pi also doesn't have a free() function so we need to reboot after each segment :( or perhaps only malloc a limited amount of memory. my approach in this file is the latter.
*/
#include "rpi.h"

#include "cycle-count.h"
#include "tokenizer.c"
#include "transformer.h"
#include "sampler.c"
#include "util.h"
#include "quant.h"
#include "forwardq.c"

#define INP_PROMPT "hello world"  // input string to run inference on

// default parameters
char *checkpoint_path = 0x0;  // e.g. out/model.bin
char *tokenizer_path = "tokenizer.bin";
float temperature = 1.0f;   // 0.0 = greedy deterministic. 1.0 = original. don't set higher
float topp = 0.9f;          // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
int steps = 8;            // number of steps to run for
char *prompt = 0x0;        // prompt string
unsigned long long rng_seed = 0; // seed rng with time by default
char *mode = "generate";    // generate|chat
char *system_prompt = 0x0; // the (optional) system prompt to use in chat mode

int  generate(Config * p, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps) {
    // encode the (string) prompt into tokens sequence
    // int num_prompt_tokens = 0;
    // int* prompt_tokens = (int*)kmalloc((strlen(prompt)+3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS
    // encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    // if (num_prompt_tokens < 1) {
    //   panic("expected at least one prompt token");
    // }

    // pre-tokenized "hello world" for speed
    int prompt_tokens[] = {1, 22172, 3186};
    int num_prompt_tokens = 3;

    // start the main loop
    long start = 0;  // used to time our code, only initialized after first iteration
    int next;        // will store the next token in the sequence
    int token = prompt_tokens[0]; // kick off with the first token in the prompt
    int pos = 0;     // position in the sequence
    while (pos < steps) {

        // forward the transformer to get logits for the next token
        float* logits = forwardq(p, token, pos);

        // advance the state state machine
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
        safe_printk(piece); // same as printk("%s", piece), but skips "unsafe" bytes
        uart_flush_tx();
        token = next;

        // init the timer here because the first iteration can be slower
        if (start == 0) { start = cycle_cnt_read();}
    }
    printk("\n");

    // report achieved tok/s (pos-1 because the timer starts after first iteration)
    if (pos > 0) {
        long end = cycle_cnt_read();
        printk("generation took %d cycles\n", end - start);
        printk("num tokens generated: %d\n", pos);
        printk("tok/cycle: %f\n", (pos-1) / (double)(end-start));
    }
    return pos;
}


void notmain() {
    rpi_init();
    cycle_cnt_init();

    // init filesys and cd to model dir
    fat32_fs_t fs;
    pi_dirent_t root;
    pi_dirent_t *w_dir = init_fs_cd_model(&fs, &root);

    // load config
    int *config_data = (int *)read_file(&fs, w_dir, "CONFIG.BIN");
    // first type bytes are magic number (0xak80) and version (2)
    assert(config_data[0] == 0x616B3830);
    assert(config_data[1] == 2);
    Config *config = (Config *)(config_data + 2);
    // config_data[9] := shared_classifier (we assume 1)
    // config_data[10] := group_size (we assume 64)

    // load tokenizer
    void *tokenizer_data = read_file(&fs, w_dir, "TOKEN~37.BIN");
    Tokenizer tokenizer;
    printk("Loading tokenizer...\n");
    build_tokenizer(&tokenizer, tokenizer_data, config->vocab_size);
    printk("Tokenizer loaded. max_token_length: %d\n", tokenizer.max_token_length);

    // temperature, topp, rng_seed defined in params.h
    Sampler sampler;
    printk("Building sampler...\n");
    build_sampler(&sampler, config->vocab_size, temperature, topp, rng_seed);
    printk("Sampler built.\n");

    // steps defined in params.h
    generate(config, &tokenizer, &sampler, INP_PROMPT, steps);

    clean_reboot();
}