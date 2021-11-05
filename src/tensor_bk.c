#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tensor.h"

tensor tensor_initialization(unsigned int size, bool requires_grad){
    float *data = (float *)calloc(size, sizeof(float));
    float *grad = (float *)calloc(size, sizeof(float));
    tensor t = {data, grad, size, requires_grad};
    return t;
}

tensor tensor_zeros(u_int32_t *shape, u_int32_t dim){
    u_int32_t size = 0;
    int i=0;
    for(i=0;i<dim;i++){
        size += shape[dim];
    }
    float *data = (float *)calloc(size, sizeof(float));
    float *grad = (float *)calloc(size, sizeof(float));
    bool requires_grad = true;
    return {data,grad,shape,dim,requires_grad};
}


void print_tensor(tensor t)
{
    int i;
    printf("Tensor size is: %d\n",t.size);
    printf(" __");
    printf("Tensor data are: \n");
    for(i=0;i<t.size;i++){
        printf("%15.7f ",t.data[i]);
    }
    printf(" \n");
    printf("Tensor gradients are: \n");
    for(i=0;i<t.size;i++){
        printf("%15.7f ",t.grad[i]);
    }
    printf(" \n");
}

// void print_tensor(tensor t)
// {
//     int i, j;
//     printf("Tensor size is: %d, %d\n",t.m.rows, t.m.cols);
//     printf(" __");
//     for(j = 0; j < 16*t.m.rows-1; ++j) printf(" ");
//     printf("__ \n");

//     printf("|  ");
//     for(j = 0; j < 16*t.m.cols-1; ++j) printf(" ");
//     printf("  |\n");

//     for(i = 0; i < t.m.rows; ++i){
//         printf("|  ");
//         for(j = 0; j < t.m.cols; ++j){
//             printf("%15.7f ", t.m.vals[i][j]);
//         }
//         printf(" |\n");
//     }
//     printf("|__");
//     for(j = 0; j < 16*t.m.cols-1; ++j) printf(" ");
//     printf("__|\n");
// }

