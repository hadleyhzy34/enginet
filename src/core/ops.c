#include <stdio.h>
#include <stdlib.h>
#include <string.h>
//#include "cuda.h"
#include "cublas_v2.h"
#include "ops.h"

#define SWAP(a,b) {float temp=0;temp=a;a=b;b=temp;}

/*-----------general matrix summation-------*/
float gems(float *a, size_t size){
    int i;
    float sum = 0;
    #pragma omp parallel for
    for(i = 0; i < size; i++){
        sum += a[i];
    }
    return sum;
}

/*----------general matrix additon--------*/
float* gema(float *a, float *b, size_t size){
    float *data = (float*)calloc(size, sizeof(float));
    int i;
    #pragma omp parallel for
    for(i = 0; i < size; i++)
        data[i] = a[i] + b[i];
    return data;
}

/*-------matrix transpose------*/
void transpose(float *a, size_t r, size_t c)
{
    /*
        Args:
            r: original rows
            c: original columns
        return:
            inplace transpose matrix a
    */
    register float *temp = (float*)calloc(r*c, sizeof(float));
    int i,j;
    for(i=0;i<r;i++){
        for(j=0;j<c;j++){
            temp[j*r + i] = a[i*c + j];
            //printf("%ld: %f\n", j*r+i, temp[j*r + i]);
        }
    }
    memcpy(a, temp, r*c*sizeof(float));
    free(temp);
}

/*-----------general matrix multiplicaton-------*/
void gemm(int TA, int TB, int M, int N, int K, float Alpha,                     
         float *A, int lda,
         float *B, int ldb,
         float Beta,
         float *C, int ldc){
     /*                                                                          
         C = Alpha * A * B + Beta * C                                            
         Args:                                                                   
             TA: A transpose or not                                              
             TB: B transpose or not                                              
             M: A.rows, C.rows                                                   
             N: B.columns, C.columns                                             
             K: A.columns, B.rows                                                
             Alpha: scalar                                                       
             Beta: scalar                                                        
             A: matrix in                                                        
             B: matrix in                                                        
             C: matrix out                                                       
             lda: leading dimension of matrix A                                  
             ldb: leading dimension of matrix B                                  
             ldc: leading dimension of matrix C                                  
     */                                                                          
     if(Beta==0){                                                                
         if(!TA&&TB){                                                            
             gemm_nt(M,N,K,Alpha,A,lda,B,ldb,C,ldc);                             
         }                                                                       
     }                                                                           
}                                                                               
                                                                                 
void gemm_nt(int M, int N, int K, float Alpha,
        float *A, int lda,                                                      
        float *B, int ldb,                                                      
        float *C, int ldc){                                                     
    int i,j,k;                                                                  
    #pragma omp parallel for                                                    
    for(i = 0; i < M; ++i){    //rows of matrix A
        for(k = 0; k < K; ++k){ //inter dimension
            register float temp = Alpha * A[i*lda+k];
            for(j = 0; j < N; ++j){ //columns of matrix B
                C[i*ldc+j] += temp * B[k*ldb+j];                                
            }
        }
    }
}

void print_matrix(float *a, size_t r, size_t c){
    int i,j;  
    printf("\n-----------------------\n");
    for(i=0;i<r;i++){
        for(j=0;j<c;j++){
            printf("%15.7f ",a[i*c+j]);
        }
        printf("\n-----------------------\n");
    }
}

#ifdef GPU
#include <math.h>

void gemm_gpu(int TA, int TB, int M, int N, int K, float Alpha,
        float *A, int lda,
        float *B, int ldb,
        float Beta,
        float *C, int ldc)
{
    //cublasHandle_t handle = blas_handle();
    //print_matrix(A,M,K);
    //print_matrix(B,K,N);
    //printf("TA: %d, TB: %d, M: %d, N: %d, K: %d, Alpha: %f, Beta: %f",TA,TB,M,N,K,Alpha,Beta);

    /*
    cudaError_t status = cublasSgemm(handle, (TB ? CUBLAS_OP_T : CUBLAS_OP_N),
            (TA ? CUBLAS_OP_T : CUBLAS_OP_N),N,M,K,&Alpha,B,ldb,A,lda,&Beta,C,ldc);
    */
    cublasHandle_t handle;
    cublasCreate(&handle);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeof(float)*M*K);
    cudaMalloc(&d_B, sizeof(float)*K*N);
    cudaMalloc(&d_C, sizeof(float)*M*N);
    cudaMemcpy(d_A,A,sizeof(float)*M*K,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,B,sizeof(float)*K*N,cudaMemcpyHostToDevice);
    cudaMemcpy(d_C,C,sizeof(float)*M*N,cudaMemcpyHostToDevice);

    //cudaMemcpy(C,d_C,sizeof(float)*9,cudaMemcpyDeviceToHost);
    //printf("matrix before gpu");
    //print_matrix(C,M,N);
    cudaError_t status = cublasSgemm(handle,
                                    (TA ? CUBLAS_OP_T : CUBLAS_OP_N),
                                    (TB ? CUBLAS_OP_T : CUBLAS_OP_N),
                                    M,N,K,&Alpha,
                                    d_A,lda,
                                    d_B,ldb,
                                    &Beta,
                                    d_C,ldc);
    cudaMemcpy(C,d_C,sizeof(float)*M*N,cudaMemcpyDeviceToHost);
    //printf("matrix after gpu");
    //print_matrix(C,M,N);
}

void mmul_gpu(int M, int N, int K,
              float *A, float *B, float *C)
{
    cublasHandle_t handle;
    cublasCreate(&handle);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeof(float)*M*K);
    cudaMalloc(&d_B, sizeof(float)*K*N);
    cudaMalloc(&d_C, sizeof(float)*M*N);
    cudaMemcpy(d_A,A,sizeof(float)*M*K,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,B,sizeof(float)*K*N,cudaMemcpyHostToDevice);
    cudaMemcpy(d_C,C,sizeof(float)*M*N,cudaMemcpyHostToDevice);
    
    float Alpha = 1.f;
    float Beta = 0.f;
    cudaError_t status = cublasSgemm(handle,
                                    CUBLAS_OP_N,
                                    CUBLAS_OP_N,
                                    N,M,K,&Alpha,
                                    d_B,N,
                                    d_A,K,
                                    &Beta,
                                    d_C,N);
    cudaMemcpy(C,d_C,sizeof(float)*M*N,cudaMemcpyDeviceToHost);
}

/*--------------------matrix summation------------------*/
float gems_gpu(float *A, size_t size)
{
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    float *res = (float*)malloc(sizeof(float)); 
    
    float *d_A;
    cudaMalloc(&d_A, sizeof(float)*size);
    cudaMemcpy(d_A,A,sizeof(float)*size,cudaMemcpyHostToDevice);
    
    int incx = 1;
    cudaError_t status = cublasSasum(handle,
                                    size,
                                    d_A,
                                    incx,
                                    res);
    return res[0];    
}

/*-----------------matrix addition------------------*/
void gema_gpu(float *A, float *B, float *C, size_t N)
{
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    memcpy(C,B,N * sizeof(float));

    float *d_A, *d_C;
    cudaMalloc(&d_A, sizeof(float)*N);
    cudaMalloc(&d_C, sizeof(float)*N);
    cudaMemcpy(d_A,A,sizeof(float)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C,C,sizeof(float)*N, cudaMemcpyHostToDevice);
    
    float alpha = 1.0;
    cublasSaxpy_v2(handle, N, &alpha, d_A, 1, d_C, 1); //vector addition
    cudaMemcpy(C,d_C,sizeof(float)*N, cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_C);
    cublasDestroy(handle);
}


/*-----------------matrix transpose----------------*/

void transpose_gpu(float *a, size_t r, size_t c)
{
    cublasHandle_t handle;
    cublasCreate(&handle);
    
     
    float const alpha = 1.f;
    float const beta = 0.f;

    float *d_A, *d_B;
    cudaMemcpy(d_A,a,sizeof(float)*r*c,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,a,sizeof(float)*r*c,cudaMemcpyHostToDevice);

    cublasSgeam( handle, CUBLAS_OP_T, CUBLAS_OP_N, 
            r, c, &alpha, d_A, c, &beta, d_B, r, d_B, r );
    cudaMemcpy(a,d_A,sizeof(float)*r*c,cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cublasDestroy(handle);
}
#endif
