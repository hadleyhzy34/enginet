#ifndef GEMM_H
#define GEMM_H
#include <stdio.h>
#include <stdbool.h>
#include "enginet.h"

void gemm(int TA, int TB, int M, int N, int K, float Alpha,
        float *A, int lda,
        float *B, int ldb,
        float Beta,
        float *C, int ldc);

void gemm_nt(int M, int N, int K, float Alpha,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc);
#endif
