#include "prune.h"

float l1_norm(float val) {
    return fabsf(val);
}

// passed into qsort
int compare_floats(const void* a, const void* b) {
    float x = *(float*)a;
    float y = *(float*)b;
    if (l1_norm(x) < l1_norm(y)) return -1;
    if (l1_norm(x) > l1_norm(y)) return 1;
    return 0;
}

void prune_matrix(float* matrix, int size, float prune_ratio) {
    int num_to_prune = (int)(size * prune_ratio);

    // cpy matrix to sort
    float* sorted_matrix = (float*)malloc(size * sizeof(float));
    memcpy(sorted_matrix, matrix, size * sizeof(float));

    // sort by l1 norm in ascending order
    qsort(sorted_matrix, size, sizeof(float), compare_floats);

    float threshold = l1_norm(sorted_matrix[num_to_prune - 1]);

    for (int i = 0; i < size; i++) {
        if (l1_norm(matrix[i]) <= threshold) {
            matrix[i] = 0.0f;
        }
    }
    free(sorted_matrix);
    printf("Pruned %d weights\n", num_to_prune);
}

float dequantize_value(int8_t q, float s) {
    return q * s;
}

void prune_qmatrix(QuantizedTensor* qmatrix, int n, int size_each, float prune_ratio) {
    int num_to_prune = (int)(size_each * 
    prune_ratio);
    if (num_to_prune == 0) {
        return;
    }

    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < n; i++) {
        float* dequantized = (float*)malloc(size_each * sizeof(float));
        for (int j = 0; j < size_each; j++) {
            dequantized[j] = dequantize_value(qmatrix[i].q[j], qmatrix[i].s[j / GS]);
        }
        
        // make a copy for sorting
        float* sorted_dequantized = (float*)malloc(size_each * sizeof(float));
        memcpy(sorted_dequantized, dequantized, size_each * sizeof(float));

        // sort by l1 norm in ascending order
        qsort(sorted_dequantized, size_each, sizeof(float), compare_floats);

        float threshold = l1_norm(sorted_dequantized[num_to_prune - 1]);

        free(sorted_dequantized);

        int8_t *new_q = (int8_t*)malloc(size_each * sizeof(int8_t));

        for (int j = 0; j < size_each; j++) {
            if (l1_norm(dequantized[j]) <= threshold) {
                new_q[j] = 0;
            } else {
                new_q[j] = qmatrix[i].q[j];
            }
        }
        qmatrix[i].q = new_q;

        free(dequantized);
    }
    printf("Pruned %d weights\n", num_to_prune * n);
}