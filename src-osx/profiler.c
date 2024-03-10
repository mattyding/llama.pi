// memory profiler that estimates the amount of RAM to run a model by tracing malloc calls
// fancier version using wrapper fns and dlsym to resolve
// i'm realizing in hindsight that C has lots of good memory profiler tools already, alas...
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>

enum {VERBOSE = 0};

static void *(*orig_malloc)(size_t size) = NULL;
static void *(*orig_calloc)(size_t nmemb, size_t size) = NULL;

static long unsigned mem = 0;

// malloc wrapper
void *malloc(size_t size) {
    // resolve the original malloc function if not already done
    if (orig_malloc == NULL) {
        orig_malloc = dlsym(RTLD_NEXT, "malloc");
    }

    // Allocate memory using the original malloc
    void *ptr = orig_malloc(size);

    if (VERBOSE) printf("malloc call from %x for %zu bytes\n", *(unsigned int *)__builtin_return_address(0), size);
    mem += size;

    return ptr;
}

// calloc wrapper
void *calloc(size_t nmemb, size_t size) {
    // resolve the original calloc function if not already done
    if (orig_calloc == NULL) {
        orig_calloc = dlsym(RTLD_NEXT, "calloc");
    }

    // Allocate memory using the original calloc
    void *ptr = orig_calloc(nmemb, size);

    if (VERBOSE) printf("calloc call from %x for %zu bytes\n", *(unsigned int *)__builtin_return_address(0), nmemb * size);
    mem += nmemb * size;

    return ptr;
}

// invoke this within main function
void print_memory_usage() {
    printf("minimum memory usage:\n");
    printf("total: %lu bytes (%lu MB)\n", mem, mem / (1024 * 1024));
}