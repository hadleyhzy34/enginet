#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gemm.h"

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
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float temp = Alpha * A[i*lda+k];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += temp * B[k*ldb+j];
            }
        }
    }
}
