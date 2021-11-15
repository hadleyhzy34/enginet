#include <stdio.h>
#include <stdlib.h>
#include "math.h"
#include "cublas_v2.h"
#include "enginet.h"
//nvcc -lcublas cublas.c -o cublas.out

int main(int argc, char* argv[])
{
int i,j,k,index;

// Linear dimension of matrices
int dim = 20;
int batch_count = 10*10*10*10*10*1;
// Allocate host storage for batch_count A,B,C square matrices
float* h_A = malloc(sizeof(float) * dim * dim * batch_count);
float* h_B = malloc(sizeof(float) * dim * dim * batch_count);
float* h_C = malloc(sizeof(float) * dim * dim * batch_count);
    for(k=0; k<batch_count; k++) {
        for(j=0; j<dim; j++) {
                for(i=0; i<dim; i++) {
                index = i*dim + j + k*dim*dim;
                  h_A[index] = index*index + 0.0f;
                  h_B[index] = index + 1.0f;
                  h_C[index] = 0.0f;
        }
    }
}


printf("------------------matrix A is---------------------\n");
print_matrix(h_A, dim, dim);
printf("------------------matrix C is---------------------\n");
print_matrix(h_C, dim, dim);

float *d_A, *d_B, *d_C;
cudaMalloc(&d_A, sizeof(float) * dim * dim * batch_count);
cudaMalloc(&d_B, sizeof(float) * dim * dim * batch_count);
cudaMalloc(&d_C, sizeof(float) * dim * dim * batch_count);
cudaMemcpy(d_A,h_A,sizeof(float) * dim * dim * batch_count,cudaMemcpyHostToDevice);
cudaMemcpy(d_B,h_B,sizeof(float) * dim * dim * batch_count,cudaMemcpyHostToDevice);
cudaMemcpy(d_C,h_C,sizeof(float) * dim * dim * batch_count,cudaMemcpyHostToDevice);

/*
cudaMemcpy(h_A,d_A,sizeof(float) * dim * dim * batch_count,cudaMemcpyDeviceToHost);
cudaMemcpy(h_B,d_B,sizeof(float) * dim * dim * batch_count,cudaMemcpyDeviceToHost);
cudaMemcpy(h_C,d_C,sizeof(float) * dim * dim * batch_count,cudaMemcpyDeviceToHost);
*/

cublasHandle_t handle;
cublasCreate(&handle);

// Do the actual multiplication
float time_cuda_event;
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop) ;
cudaEventRecord(start, 0);
float alpha = 1.0f;  float beta = 1.0f;
cublasSgemmStridedBatched(handle,
                              CUBLAS_OP_N,
                              CUBLAS_OP_N,
                              dim, dim, dim,
                              &alpha,
                              (const float*)d_A, dim,
                              dim*dim,
                              (const float*)d_B, dim,
                              dim*dim,
                              &beta,
                              d_C, dim,
                              dim*dim,
                              batch_count);
( cudaEventRecord(stop, 0) );
( cudaEventSynchronize(stop) );
( cudaEventElapsedTime(&time_cuda_event, start, stop) );
printf("Time :  %3.1f ms \n", time_cuda_event);

cudaMemcpy(h_C,d_C,sizeof(float) * dim * dim * batch_count,cudaMemcpyDeviceToHost);

printf("------------------matrix A is---------------------\n");
print_matrix(h_A, dim, dim);
printf("------------------matrix C is---------------------\n");
print_matrix(h_C, dim, dim);
// Destroy the handle
cublasDestroy(handle);


cudaFree(d_A);
cudaFree(d_B);
cudaFree(d_C);
free(h_A);
free(h_B);
free(h_C);

return 0;
}
