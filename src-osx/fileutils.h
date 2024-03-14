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

// filepaths
static char *config_fp32_path = "../models/7b-f32/config.bin";
static char *config_q80_path = "../models/7b-q80/config.bin";
static char *tokenizer_path = "../models/tokenizer.bin";

static char *full_weights_fp32_path = "../models/7b-f32/full_weights.bin";
static char *full_weights_q80_path = "../models/7b-q80/full_weights.bin";

// fns
// magic and version are used to check right filepath
// magic is "ak32" in ascii for FP32 and "ak80" in ascii for Q80
// version is 1 for FP32 and 2 for Q80
void read_config(char* config_path, Config* config, int magic, int version);
void load_full_weights_fp32(Config *p, TransformerWeights *w);
void load_full_weights_q80(Config *p, qTransformerWeights *w);

#endif