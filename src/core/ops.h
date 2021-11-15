#ifndef OPS_H
#define OPS_H
#include <stdio.h>
#include <stdbool.h>
#include "enginet.h"
#include "tensor.h"

#ifdef GPU
float gems_gpu(float *a, size_t size);
void gema_gpu(float *a, float *b, float *c, size_t size);
void transpose_gpu(float *a, size_t r, size_t c);
void gemm_gpu(int TA, int TB, int M, int N, int K, float Alpha,
            float *A, int lda,
            float *B, int ldb,
            float Beta,
            float *C, int ldc);
void mmul_gpu(int M, int N, int K, float *A, float *B, float *C);
#endif

float  gems(float *a, size_t size);
float* gema(float *a, float *b, size_t size);
void transpose(float *a, size_t r, size_t c);

void gemm(int TA, int TB, int M, int N, int K, float Alpha,
          float *A, int lda,
          float *B, int ldb,
          float Beta,
          float *C, int ldc);

void gemm_nt(int M, int N, int K, float Alpha,
             float *A, int lda,
             float *B, int ldb,
             float *C, int ldc);

void print_matrix(float *a, size_t r, size_t c);

#endif


