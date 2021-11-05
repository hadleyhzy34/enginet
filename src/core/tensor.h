#ifndef TENSOR_H
#define TENSOR_H
#include <stdio.h>
#include <stdbool.h>
#include "enginet.h"

tensor tensor_zeros(int *shape, size_t dim);
tensor tensor_ones(int *shape, size_t dim);

void reshape_tensor(tensor *t, int *shape, size_t dim);
void print_tensor(tensor t);

#endif
