#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tensor.h"

tensor tensor_zeros(int *shape, size_t dim){
    size_t size = dim?1:0;
    int *c_shape = (int *)malloc(dim*sizeof(int));
    memcpy(c_shape,shape,dim*sizeof(int));
    int i;
    for(i=0;i<dim;i++){
        size *= shape[i];//total number of elements inside tensor
    }
    float *data = (float *)calloc(size, sizeof(float));
    float *grad = (float *)calloc(size, sizeof(float));
    bool requires_grad = true; //default gradient is required

    return (tensor){data,grad,c_shape,dim,requires_grad};
}

tensor tensor_ones(int *shape, size_t dim){
    size_t size = dim?1:0;
    int *c_shape = (int *)malloc(dim*sizeof(int));
    memcpy(c_shape,shape,dim*sizeof(int));

    int i=0;
    for(i=0;i<dim;i++){
        size *= shape[i];
        //printf("current size and shape are: %ld, %d",size, shape[i]);
    }
    float *data = (float *)calloc(size, sizeof(float));
    for(i=0;i<size;i++){
        data[i] = 1;
    }
    float *grad = (float *)calloc(size, sizeof(float));
    bool requires_grad = true; //default gradient is required
    return (tensor){data,grad,c_shape,dim,requires_grad};
}

void print_tensor(tensor t)
{
    int i;
    printf("Tensor dimension is: %ld\n",t.dim);
    size_t size=1;
 
    /*---------------tensor shape+size---------------*/
    printf("current tensor shape is: ");
    for(i=0;i<t.dim;i++){
        printf("%d ", t.shape[i]);
        size *= t.shape[i];
    }
 
    /*---------------tensor data---------------*/
    printf("\ntensor size is: %ld\n", size);
    for(i=0;i<size;i++){
        printf("%15.7f ", t.data[i]);
    }
    printf(" \n");
 
    /*---------tensor gradients--------*/
    printf("Tensor gradients are: \n");
    for(i=0;i<size;i++){
        printf("%15.7f ",t.grad[i]);
    }
    printf(" \n");

    /*-------------tensor require grad---------*/
    printf("tensor requires_grad: %s\n", t.requires_grad?"true":"false");
}

void reshape_tensor(tensor *t, int* n_shape, size_t dim)
{
    t->shape = realloc(t->shape, dim*sizeof(int));
    /*free(t.shape);
    t.shape = (int *)malloc(dim*sizeof(int));*/
    memcpy(t->shape,n_shape,dim*sizeof(int));
    t->dim = dim;
    //print_tensor(&t);
}
