#ifndef ENGINET_API
#define ENGINET_API
#include <stdlib.h>
#include <stdbool.h>

/*vector, flatten matrix */
typedef struct{
    int rows,cols;
    float *vals;
}vector;

/* 2d matrix */
typedef struct{
    int rows,cols;
    float **vals;
} matrix;

// /*tensor*/
// typedef struct{
//     matrix grad;
//     bool requires_grad;
//     bool is_leaf;
//     matrix m;
// } tensor;

//tensor struct
typedef struct{
    float *data; //tensor data
    float *grad; //gradients, alpha(loss)/ahpha(x)
    int size;
    bool requires_grad;
}tensor;

/*--------------------------------fc_layer-------------------------------*/
//full connection layer definition
typedef struct{
    tensor input;  //input data
    tensor output;  //output data
    int channels_in;  //input size
    int channels_out;  //output size
    float *weights;  //fc layer weights
    float *bias; //fc layer bias
    float lr;  //learning rate for updating fc layer
} fc_layer;

//fc layer function
fc_layer fc_layer_initialization(tensor input, int channels_in, int channels_out);
void forward_fc_layer(fc_layer l);
void backward_fc_layer(fc_layer l);
void zero_grad_fc_layer(fc_layer l);

/*--------------------------------activation-------------------------------*/
//activation definition
typedef enum{RELU, TANH}ACTIVATION;

//activation function
ACTIVATION get_activation(char *s);
float activate(float x, ACTIVATION a);
void activate_array(float *x, const int size, const ACTIVATION a);
float activate_gradient(float x, ACTIVATION a);
void activate_gradient_array(const float *x, const int size, const ACTIVATION a, float *delta);

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

/*loss function*/
float mean_square_error(tensor output, tensor label);
void zero_grad_mse(tensor output);

#endif
