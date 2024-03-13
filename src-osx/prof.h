// file containing profiler functions
#ifndef __PROF_H__
#define __PROF_H__
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <dlfcn.h>
#include <stdio.h>
#include <x86intrin.h>
#include <time.h>

// --- time profiler ---
static unsigned long long start_tsc, end_tsc;
static unsigned long long (*orig_func)(void);
static struct timespec start_time, end_time;

void start_timer(void);
void stop_timer(void);
void reset_timer(void);

void print_time_elapsed(char * msg, int num_tokens);

// --- memory profiler ---
static long unsigned mem = 0;

void reset_mem_prof(void);
void print_mem_prof(char *msg);

#endif