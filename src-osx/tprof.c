#include "prof.h"

void start_timer(void) {
    start_tsc = __rdtsc();
    clock_gettime(CLOCK_MONOTONIC, &start_time);
}

void stop_timer(void) {
    end_tsc = __rdtsc();
    clock_gettime(CLOCK_MONOTONIC, &end_time);
}

void reset_timer(void) {
    start_tsc = 0;
    end_tsc = 0;
    start_time.tv_sec = 0;
    start_time.tv_nsec = 0;
    end_time.tv_sec = 0;
    end_time.tv_nsec = 0;
}

void print_time_elapsed(char * msg, int num_tokens) {
    unsigned long long elapsed_cycles = end_tsc - start_tsc;
    long long elapsed_us = (end_time.tv_sec - start_time.tv_sec) * 1000000LL +
                          (end_time.tv_nsec - start_time.tv_nsec) / 1000;
    if (msg) {
        printf("%s\n", msg);
    }
    printf("\tElapsed cycles: %llu\n", elapsed_cycles);
    printf("\tElapsed time: %lld us\n", elapsed_us);
    if (num_tokens > 0) {
        printf("\t(cyc/token): %f\n", (float)elapsed_cycles / num_tokens);
        printf("\t(us/token): %f us\n", (float)elapsed_us / num_tokens);
    }
}