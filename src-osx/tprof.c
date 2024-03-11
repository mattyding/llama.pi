#include <stdio.h>
#include <x86intrin.h>
#include <time.h>

static unsigned long long start_tsc, end_tsc;
static unsigned long long (*orig_func)(void);
static struct timespec start_time, end_time;

void start_timer(void) {
    start_tsc = __rdtsc();
    clock_gettime(CLOCK_MONOTONIC, &start_time);
}

void stop_timer(void) {
    end_tsc = __rdtsc();
    clock_gettime(CLOCK_MONOTONIC, &end_time);
}

void print_time_elapsed(void) {
    unsigned long long elapsed_cycles = end_tsc - start_tsc;
    long long elapsed_us = (end_time.tv_sec - start_time.tv_sec) * 1000000LL +
                          (end_time.tv_nsec - start_time.tv_nsec) / 1000;
    printf("Elapsed cycles: %llu\n", elapsed_cycles);
    printf("Elapsed time: %lld us\n", elapsed_us);
    int num_tokens = 3;
    printf("(cycs/token): %f\n", (float)elapsed_cycles / num_tokens);
    printf("(us/token): %f us\n", (float)elapsed_us / num_tokens);
}

// example usage
// #include "tprof.h"
// int main() {
//     start_timer();
//     // do something
//     stop_timer();
//     print_time_elapsed();
// }