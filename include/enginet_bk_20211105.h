#ifndef ENGINET_API
#define ENGINET_API
#include <stdlib.h>
#include <stdbool.h>

#define LR 0.01
#define EPOCHS 500
#define BATCH_SIZE 1

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

/*-------------------tensor------------------*/
typedef struct{
    float *data; //tensor data
    float *grad; //gradients, alpha(loss)/ahpha(x)
    unsigned int size; //data size for each sample data
    bool requires_grad;
}tensor;

/*tensor initialization*/
tensor tensor_initialization(unsigned int size, bool requires_grad);

typedef struct{
    tensor *t;
    const unsigned int batch_size;
}tensors;

/*-----------------fc_layer------------------*/
//full connection layer definition
// typedef struct{
//     tensor input;  //input data
//     tensor output;  //output data
//     int channels_in;  //input size
//     int channels_out;  //output size
//     float *weights;  //fc layer weights
//     float *bias; //fc layer bias
//     float lr;  //learning rate for updating fc layer
// } fc_layer;
typedef struct{
    tensor *input;
    tensor *output;
    const unsigned int batch_size;
    int in_channels;
    int out_channels;
    float *weights;
    float *bias;
}fc_layer;

//fc layer function
fc_layer fc_layer_initialization(const unsigned int batch_size, tensor *input, int in_channels, int out_channels);
void forward_fc_layer(fc_layer l);
void backward_fc_layer(fc_layer l);
void zero_grad_fc_layer(fc_layer l);

/*----------------activation-----------------*/
//activation definition
typedef enum{RELU, TANH}ACTIVATION;

//activation function
ACTIVATION get_activation(char *s);
float activate(float x, ACTIVATION a);
void activate_array(float *x, const int size, const ACTIVATION a);
void activate_tensor(tensor input, tensor output, ACTIVATION a);
float activate_gradient(float x, ACTIVATION a);
void activate_gradient_array(const float *x, const int size, const ACTIVATION a, float *delta);

//activation layer definition
typedef struct{
    const unsigned int batch_size;  //batch size
    tensor *input;  //input data
    tensor *output;  //output data
    ACTIVATION a; //activation enum
} ac_layer;

ac_layer ac_layer_initialization(const unsigned int batch_size, tensor *input, ACTIVATION a);
void forward_ac_layer(ac_layer l);
void backward_ac_layer(ac_layer l);
void zero_grad_ac_layer(ac_layer l);


/*----------------------------network-------------------------------*/
typedef struct{
    fc_layer ** layers;
    unsigned int batch_size;
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
float mean_square_error(const unsigned int batch_size, tensor *output, tensor *label);
void zero_grad_mse(tensor output);

#endif
