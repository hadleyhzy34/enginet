#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cublas_v2.h>

void creatematrix(float *out, int nx, int ny)
{
    float ctr = 0;
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            ctr = ctr + 1;
            out[j * nx + i] = ctr;
        }
    }
}

float add_arr_val(float *im, int N)
{
    float tmp = 0;
    for (int i = 0; i < N; ++i)
        tmp += im[i];

    float out = tmp;
    return out;
}

void main()
{
    // Define matrix size (using flattened array for most operations)
    int nx = 10;       // row size
    int ny = 10;       // column size
    int N = nx * ny;    // total size of flattened array
    
    // CPU section ========================================
    float *M; M = (float*)malloc(N * sizeof(float));    // create array pointer and allocate memory
    creatematrix(M, nx, ny);                            // create a test matrix of size nx * ny
    float cpu_out = add_arr_val(M, N);                  // CPU function

    // GPU and cuBLAS section ==============================
    float *d_M;
    cudaMalloc(&d_M, N * sizeof(float));
    cudaMemcpy(d_M, M, N * sizeof(float), cudaMemcpyHostToDevice);
        
    // create array of all ones, size N for dot product
    float *d_ones;
    cudaMalloc(&d_ones, N * sizeof(float));

    float *ones = (float*)malloc(N*sizeof(float)); 

    int i;
    for(i=0;i<N;i++){
        ones[i] = 1.f;
    }
    cudaMemcpy(d_ones,ones,N*sizeof(float),cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    
    float blas_out;                                                         // output on host variable
    cublasHandle_t handle;  cublasCreate(&handle);                          // initialize CUBLAS context        
    cublasSdot(handle, N, d_M, 1, d_ones, 1, &blas_out);                    // Perform cublas single-precision dot product of (d_M . d_ones)
    cudaDeviceSynchronize();    
    
    //print output from cpu and gpu sections
    printf("native output = %lf\n", cpu_out);
    printf("cublas output = %lf\n", blas_out);

    cublasDestroy(handle);
    free(M);
    cudaFree(d_M);
    cudaFree(d_ones);
}
