// memory profiler that estimates the amount of RAM to run a model by tracing malloc calls
// fancier version using wrapper fns and dlsym to resolve
// i'm realizing in hindsight that C has lots of good memory profiler tools already, alas...
#include "prof.h"

static void *(*orig_malloc)(size_t size) = NULL;
static void *(*orig_calloc)(size_t nmemb, size_t size) = NULL;

// malloc wrapper
void *malloc(size_t size) {
    // resolve the original malloc function if not already done
    if (orig_malloc == NULL) {
        orig_malloc = dlsym(RTLD_NEXT, "malloc");
    }

    // Allocate memory using the original malloc
    void *ptr = orig_malloc(size);

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

    mem += nmemb * size;

    return ptr;
}

void reset_mem_prof(void) {
    mem = 0;
}

void print_mem_prof(char *msg) {
    if (msg) {
        printf("%s\n", msg);
    }
    printf("\tMemory used: %lu bytes (%lu MB)\n", mem, mem / (1024 * 1024));
}