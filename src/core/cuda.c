#include "cuda.h"
#include <assert.h>
#include <stdlib.h>
#include <time.h>

int cuda_get_device()
{
    int n = 0;
    cudaError_t status = cudaGetDevice(&n);
    check_error(status);
    return n;
}

void check_error(cudaError_t status)
{
    cudaError_t status2 = cudaGetLastError();
    if (status != cudaSuccess)
    {
        const char *s = cudaGetErrorString(status);
        char buffer[256];
        printf("cuda error: %s\n", s);
        assert(0);
        snprintf(buffer,256,"cuda error: %s",s);
        error(buffer);
    }

    if (status2 != cudaSuccess)
    {
        const char *s = cudaGetErrorString(status);
        char buffer[256];
        printf("cuda error: %s\n", s);
        assert(0);
        snprintf(buffer,256,"cuda error: %s",s);
        error(buffer);
    }
}

cublasHandle_t blas_handle()
{
    static int init[16] = {0};
    static cublasHandle_t handle[16];
    int i = cuda_get_device();
    if(!init[i]){
        cublasCreate(&handle[i]);
        init[i] = 1;
    }
    return handle[i];
}

