#ifndef CUDA_H
#define CUDA_H

#include "enginet.h"
#include "cublas_v2.h"

#ifdef GPU
int cuda_get_device();
void check_error(cudaError_t status);
cublasHandle_t blas_handle();

#endif
#endif
