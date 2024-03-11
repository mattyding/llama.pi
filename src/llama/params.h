// want to make wrapper match notmain def so moving argv to this file

// default parameters
char *checkpoint_path = 0x0;  // e.g. out/model.bin
char *tokenizer_path = "tokenizer.bin";
float temperature = 1.0f;   // 0.0 = greedy deterministic. 1.0 = original. don't set higher
float topp = 0.9f;          // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
int steps = 256;            // number of steps to run for
char *prompt = 0x0;        // prompt string
unsigned long long rng_seed = 0; // seed rng with time by default
char *mode = "generate";    // generate|chat
char *system_prompt = 0x0; // the (optional) system prompt to use in chat mode