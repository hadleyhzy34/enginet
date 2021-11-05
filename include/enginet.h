#ifndef ENGINET_API
#define ENGINET_API
#include <stdlib.h>
#include <stdbool.h>

#define LR 0.01
#define EPOCHS 500
#define BATCH_SIZE 1

/*-------------------tensor------------------*/
typedef struct{
    float *data; //data
    float *grad; //gradients
    int *shape; //shape of tensor
    size_t dim; //dimension of tensor
    bool requires_grad;
}tensor;

/*tensor initialization*/
tensor tensor_zeros(int *shape, size_t dim);
tensor tensor_ones(int *shape, size_t dim);

/*reshape tensor*/
void reshape_tensor(tensor *t, int* n_shape, size_t dim);

/*tensor visualization*/
void print_tensor(tensor t);


/*--------------computation graph-----------*/
typedef enum{ADD, FC, Conv2d}grad_func;

typedef struct{
    grad_func grad_fn;
    tensor *weights_grad;
    tensor *bias_grad;
    tensor *grads;
    struct graph_node *next;
    struct graph_ndoe *previous;
}graph_node;

#endif
