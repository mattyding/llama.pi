#ifndef __FILEUTILS_H__
#define __FILEUTILS_H__

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>

#include "transformer.h"

// filepaths
static char *config_fp32_path = "../models/7b-f32/config.bin";
static char *tokenizer_path = "../models/tokenizer.bin";

static char *full_weights_fp32_path = "../models/7b-f32/full_weights.bin";
static char *full_weights_q80_path = "../models/7b-q80/full_weights.bin";

// fns
void read_config(char* config_path, Config* config);
void load_full_weights_fp32(Config *p, TransformerWeights *w);
void load_full_weights_q80(Config *p, qTransformerWeights *w);

#endif