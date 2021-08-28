#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tensor.h"
#include "ac_layer.h"
#include "activations.h"

// //activation definition
// typedef enum{RELU, TANH}ACTIVATION;

// //activation function
// ACTIVATION get_activation(char *s);
// float activate(float x, ACTIVATION a);
// void activate_array(float *x, const int size, const ACTIVATION a);
// float activate_gradient(float x, ACTIVATION a);
// void activate_gradient_array(const float *x, const int size, const ACTIVATION a, float *delta);

// //activation layer definition
// typedef struct{
//     tensor input;  //input data
//     tensor output;  //output data
//     ACTIVATION ac; //activation enum
//     float lr;  //learning rate for updating fc layer
// } ac_layer;

ac_layer ac_layer_initialization(tensor input, ACTIVATION a){
    /*---------------------output tensor initialization-------------------------*/
    float *output_data = (float*)calloc(input.size,sizeof(float));
    float *output_grad = (float*)calloc(input.size,sizeof(float));
    tensor output = {output_data, output_grad, input.size, true};

    ac_layer l = {input, output, a, 0.01};
    return l;
}

void forward_ac_layer(ac_layer l){
    if(l.input.size != l.output.size){
        perror("dimension not compatible for matrix activation");
        exit(EXIT_FAILURE);
    }
    unsigned int i;
    for(i = 0; i < l.input.size; i++){
        l.output.data[i] = activate(l.input.data[i], l.a);
    }
}

void backward_ac_layer(ac_layer l){
    unsigned int i;
    for(i = 0; i < l.output.size; i++){
        l.input.grad[i] += activate_gradient(l.output.data[i], l.a) * l.output.grad[i];
    }
}

//set the output layer data and gradients to be zero
void zero_grad_ac_layer(ac_layer l){
    unsigned int i;
    for(i = 0; i < l.output.size; i++){
        l.output.data[i] = 0.0;
        l.output.grad[i] = 0.0;
    }
}

