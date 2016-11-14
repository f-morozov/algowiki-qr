#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mkl.h"

void run_benchmark(size_t size, int max_threads);

int main(int argc, char** argv) {
    if(argc != 2) {
        return 1;
    }
    size_t size = (size_t)atoi(argv[1]);
    int max_threads = mkl_get_max_threads();
    srand(time(NULL));
    run_benchmark(size, max_threads);
    return 0;
}

void run_benchmark(size_t size, int max_threads) {
    size_t size_bytes = size * size * sizeof(float);
    float* matrix = (float*)mkl_malloc(size_bytes, 0);
    size_t i;
    for(i = 0; i < size * size; ++i) {
        matrix[i] = (float)rand();
    }
    
    float* work_matrix = (float*)mkl_malloc(size_bytes, 0);
    float* reflectors = (float*)mkl_malloc(size * sizeof(float), 0);
    float* real = (float*)mkl_malloc(size * sizeof(float), 0);
    float* img = (float*)mkl_malloc(size * sizeof(float), 0);
    
    int threads;
    for(threads = 1; threads <= max_threads; ++threads) {
        mkl_set_num_threads(threads);        

        memcpy(work_matrix, matrix, size_bytes); 
        double start = dsecnd();
        LAPACKE_sgehrd(LAPACK_ROW_MAJOR, size, 1, size, work_matrix, size, reflectors);
        LAPACKE_shseqr(LAPACK_ROW_MAJOR, 'E', 'N', size, 1, size, work_matrix, size, real, img, NULL, size);
        double end = dsecnd();
        printf("%d %d %lf\n", size, threads, end - start);
    }

    mkl_free(matrix);
    mkl_free(work_matrix);
    mkl_free(reflectors);
    mkl_free(real);
    mkl_free(img);
}

