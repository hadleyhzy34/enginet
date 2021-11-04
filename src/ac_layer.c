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
//     const unsigned int batch_size;  //batch size
//     tensor *input;  //input data
//     tensor *output;  //output data
//     ACTIVATION a; //activation enum
// } ac_layer;

// ac_layer ac_layer_initialization(const unsigned int batch_size, tensor *input, ACTIVATION a);
// void forward_ac_layer(ac_layer l);
// void backward_ac_layer(ac_layer l);
// void zero_grad_ac_layer(ac_layer l);

ac_layer ac_layer_initialization(const unsigned int batch_size, tensor *input, ACTIVATION a){
    unsigned int k;
    /*output batch tensor initialization*/
    tensor *output = (tensor*)malloc(batch_size*sizeof(tensor)); /*output batch memory space created*/
    for(k = 0; k < batch_size; k++){
        output[k] = tensor_initialization(input[k].size, true);
        if(input[k].size != output[k].size){                    /*check input and output dimension*/
            perror("dimension not compatible for matrix activation");
            exit(EXIT_FAILURE);
        }
    }

    ac_layer l = {batch_size, input, output, a};
    return l;
}

void forward_ac_layer(ac_layer l){
    unsigned int i,k;
    for(k = 0; k < l.batch_size; k++){ /*loop through each sample of batch*/
        for(i = 0; i < l.input[k].size; i++){
            l.output[k].data[i] = activate(l.input[k].data[i], l.a);
        }
    }
}

void backward_ac_layer(ac_layer l){
    unsigned int i,k;
    for(k = 0; k < l.batch_size; k++){
        for(i = 0; i < l.output[k].size; i++){  /*calculate gradients of layer input data*/
            l.input[k].grad[i] += activate_gradient(l.output[k].data[i], l.a) * l.output[k].grad[i];
        }
    }
}

//set the output layer data and gradients to be zero
void zero_grad_ac_layer(ac_layer l){
    unsigned int i,k;
    for(k = 0; k < l.batch_size; k++){
        for(i = 0; i < l.output[k].size; i++){  /*reset each out_channels data from each sample*/
            l.output[k].data[i] = 0.0;
            l.output[k].grad[i] = 0.0;
        }
    }
}

