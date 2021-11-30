#include "cuda.h"
#include "cuda_runtime_api.h"
#include "cublas_v2.h"
#include <stdio.h>
#include <stdlib.h>
#include "enginet.h"

int main(void)
{
    const int N = 10;
    const size_t sz = sizeof(float) * (size_t)(N);
    float *A, *I;

    float Ah[] = { 0., 1., 2., 3., 4., 5., 6., 7., 8., 9. };

    cudaMalloc((void **)&A, sz);
    cudaMemcpy(A, &Ah[0], sz, cudaMemcpyHostToDevice);

    // this creates a bit pattern for a single precision unity value
    // and uses 32-bit memset from the driver API to set the values in the
    // vector.
    const float one = 1.0f;
    const int* one_bits = (const int*)(&one);
    cudaMalloc((void **)&I, sz);
    cuMemsetD32(CUdeviceptr(I), *one_bits, N);

    float *ones = (float*)malloc(N*sizeof(float));
    cudaMemcpy(ones,I,N*sizeof(float),cudaMemcpyDeviceToHost);
    print_matrix(ones,1,N); 

    cublasHandle_t h;
    cublasCreate(&h);

    const float alpha = 5.0f;
    cublasSaxpy(h, N, &alpha, I, 1, A, 1);

    cudaMemcpy(&Ah[0], A, sz, cudaMemcpyDeviceToHost);

    
    printf("current ah vector last ele is: %.5f",Ah[9]);
    cublasDestroy(h);
    cudaDeviceReset();

    return 0;
}
