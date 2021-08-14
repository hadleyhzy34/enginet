#ifndef ENGINET_API
#define ENGINET_API
#include <stdlib.h>
#include <stdbool.h>

/* 2d matrix */
typedef struct{
    int rows,cols;
    float **vals;
} matrix;

/*tensor*/
typedef struct{
    matrix grad;
    bool requires_grad;
    bool is_leaf;
    matrix m;
} tensor;

/*fc_layer*/
typedef struct{
    tensor input;  //input data
    tensor output;  //output data
    int channels_in;
    int channels_out;
    tensor weights;  //fc layer weights
    tensor bias;  // fc layer bias
    float lr;  //learning rate for updating fc layer
} fc_layer;

/*network*/
typedef struct{
    fc_layer ** layers;
}network;

/*matrix function*/
void print_matrix(matrix m);
matrix resize_matrix(matrix m, int rows, int cols);
void free_matrix(matrix m);
void matrix_to_csv(matrix m);
void scale_matrix(matrix m, float scale);
matrix mat_add(matrix a, matrix b);
matrix mat_mul(matrix a, matrix b);
matrix mat_sub(matrix a, matrix b);
matrix mat_scal(matrix a, float scalar);

/*tensor function*/
void print_tensor(tensor t);

/*layer function*/
void forward_fc_layer(fc_layer l);
void backward_fc_layer(fc_layer l);

#endif
