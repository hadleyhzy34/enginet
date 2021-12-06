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
float gevdp_gpu(float *a, float *b, size_t size);
void geac(float *a, size_t size, float b);
void gemc(float *a, size_t size, float scalar);
void gemm_gpu(int TA, int TB, int M, int N, int K, float Alpha,
            float *A, int lda,
            float *B, int ldb,
            float Beta,
            float *C, int ldc);
void mmul_gpu(int M, int N, int K, float *A, float *B, float *C);
#endif

#ifdef NEON
float gems_arm(float *a, size_t size);
void transpose_arm(float *a, size_t r, size_t c);
float gevdp_arm(float *a, float *b, size_t size);
float* gema_arm(float *a, float *b, size_t size);
void geac_arm(float *a, size_t size, float b);
void gemc_arm(float *a, size_t size, float scalar);
#endif

float  gems(float *a, size_t size);
float* gema(float *a, float *b, size_t size);
void transpose(float *a, size_t r, size_t c);
float gevdp(float *a, float *b, size_t size);
void geac_gpu(float *a, size_t size, float b);
void gemc_gpu(float *a, size_t size, float scalar);
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


