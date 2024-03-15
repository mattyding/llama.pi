#ifndef __FILEUTILS_H__
#define __FILEUTILS_H__

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#if defined _WIN32
    #include "win.h"
#else
    #include <unistd.h>
    #include <sys/mman.h>
#endif

#include "transformer.h"

static int shared_classifier = 0;


// pruning experiments
// i think buggy atm. just run in python (exp-prune.py)
static int use_prune = 0;   // 1 if using pruned weights
static char *prune_path = "../models/7b-q80/prune/bitvec_l1_unstructured_0.2.bin";    // bitvector

// filepaths
static char *config_fp32_path = "../models/7b-f32/config.bin";
static char *config_q80_path = "../models/7b-q80/config.bin";
static char *tokenizer_path = "../models/tokenizer.bin";

static char *full_weights_fp32_path = "../models/7b-f32/full_weights.bin";
static char *full_weights_q80_path = "../models/7b-q80/full_weights.bin";

static char *tok_embed_q80_path = "../models/7b-q80/tok_emb.bin";
static char *rms_final_q80_path = "../models/7b-q80/norm.bin";
static char *wcls_q80_path = "../models/7b-q80/output.bin";

// fns
// magic and version are used to check right filepath
// magic is "ak32" in ascii for FP32 and "ak80" in ascii for Q80
// version is 1 for FP32 and 2 for Q80
void read_config(char* config_path, Config* config, int magic, int version);
void load_full_weights_fp32(Config *p, TransformerWeights *w);
void load_full_weights_q80(Config *p, qTransformerWeights *w);

void load_token_embedding_table(Config *p, qLayerWeights *w);
void load_rms_final_weight(Config *p, qLayerWeights *w);
void load_wcls(Config *p, qLayerWeights *w);

#endif