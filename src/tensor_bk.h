#ifndef TENSOR_H
#define TENSOR_H
#include <stdio.h>
#include <stdbool.h>
#include "enginet.h"


void print_tensor(tensor t);
tensor tensor_initialization(unsigned int size, bool requires_grad);

#endif
