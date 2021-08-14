#ifndef TENSOR_H
#define TENSOR_H
#include <stdio.h>
#include <stdbool.h>
#include "enginet.h"

// /*2d tensor*/
// typedef struct{
//     float grad;
//     bool is_leaf;
//     float ** vals;
//     int rows,cols;
//     bool requires_grad;
// } tensor;

void print_tensor(tensor t);

#endif
