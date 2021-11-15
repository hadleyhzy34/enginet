#ifndef ENGINET_API
#define ENGINET_API
#include <stdlib.h>
#include <stdbool.h>



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

/*---------------matrix operation------------*/
float  gems(float *a, size_t size);
float* gema(float *a, float *b, size_t size);
void transpose(float *a, size_t r, size_t c);
void print_matrix(float *a, size_t r, size_t c);

/*------------------gemm--------------------*/
void gemm(int TA, int TB, int M, int N, int K, float Alpha,
        float *A, int lda,
        float *B, int ldb,
        float Beta,
        float *C, int ldc);

void gemm_nt(int M, int N, int K, float Alpha,
        float *A, int lda,
        float *B, int ldb,
        float *C, int ldc);

#ifdef GPU
float gems_gpu(float *a, size_t size);
void gema_gpu(float *a, float *b, float *c, size_t size);
void transpose_gpu(float *a, size_t r, size_t c);
void gemm_gpu(int TA, int TB, int M, int N, int K, float Alpha,
        float *A, int lda,
        float *B, int ldb,
        float Beta,
        float *C, int ldc);
#endif

/*--------------computation graph-----------*/
typedef struct graph_node{
    //tensor *input;
    //tensor *output;
    void (*grad_fn)(tensor *input, tensor *output);
    struct graph_node *parent;
}graph_node;

/*------------tensor operation----------*/
graph_node tensor_add(tensor *a, tensor *b, graph_node *parent);

/*---------------fc layer------------------*/
typedef struct{
    //tensor *input;
    tensor *output;
    float *weights;
    float *bias;
    size_t in_channels;
    size_t out_channels;
    void (*foward)(tensor *input);
}fc_layer;

fc_layer fc(size_t in_channels, size_t out_channels); //initialize fc layer
void forward_fc(fc_layer l, struct graph_node parent); //forward pass
void backward_fc(fc_layer l); //backward pass

#endif
