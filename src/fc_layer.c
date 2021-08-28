#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matrix.h"
#include "tensor.h"
#include "fc_layer.h"

// /*fc_layer*/
// typedef struct{
//     tensor input;  //input data
//     tensor output;  //output data
//     int channels_in;  //input size
//     int channels_out;  //output size
//     float *weights;  //fc layer weights
//     float *bias; //fc layer bias
//     float lr;  //learning rate for updating fc layer
// } fc_layer;

fc_layer fc_layer_initialization(tensor input, int channels_in, int channels_out)
{
    if(input.size != channels_in){
        perror("dimension not compatible for matrix addition");
        exit(EXIT_FAILURE);
    }
    /*------------------------weights initialization----------------------------*/
    float *weights = (float *)malloc(channels_in*channels_out*sizeof(float));
    for(int i = 0; i < channels_in * channels_out; i++){
        weights[i] = ((float)rand()/RAND_MAX * 2) - 1;  //size:channels_in * channels_out
    }
    /*---------------------------bias initialization----------------------------*/
    float *bias = (float*)calloc(channels_out,sizeof(float)); //size: channels_out * 1

    /*---------------------output tensor initialization-------------------------*/
    float *output_data = (float*)calloc(channels_out,sizeof(float));
    float *output_grad = (float*)calloc(channels_out,sizeof(float));
    tensor output = {output_data, output_grad, channels_out, true};

    //debug
    // print_tensor(output);
    // fc_layer l = {output, channels_in, channels_out, weights, bias, 0.01};
    fc_layer l = {input,output,channels_in,channels_out,weights,bias,0.0001};
    return l;
}

void forward_fc_layer(fc_layer l){
    //fc layer forward operation
    int i,j;
    // printf("layer weights data is: \n");
    // for(int i=0;i<l.channels_in*l.channels_out;i++){
    //     printf("%15.7f ",l.weights[i]);
    // }
    // printf("\n");
    for(j=0;j<l.channels_out;j++){
        l.output.data[j] += l.bias[j];
        for(i=0;i<l.channels_in;i++){
            l.output.data[j] += l.input.data[i] * l.weights[i*l.channels_out+j];
            // printf("i: %d, j: %d, l.input: %f, l.weights: %f, l.output: %f\n", i,j,l.input.data[i],l.weights[i*l.channels_out+j],l.output.data[j]);
        }
    }
    // printf("tensor gradients for output is: \n");
    // print_tensor(l.output);
    // printf("\n");
}

void backward_fc_layer(fc_layer l){
    /*---------------gradients calculation---------------------------*/
    /* gradient of inputs
    alpha(loss)/alpha(input) = output_gradients * alpha(output)/alpha(input)
    */
    int i,j;
    for(i=0;i<l.channels_in;i++){
        for(j=0;j<l.channels_out;j++){
            l.input.grad[i]+=l.output.grad[j]*l.weights[i*l.channels_out+j];
        }
    }

    // printf("tensor gradients for output is: \n");
    // print_tensor(l.output);
    // printf("\n");

    //update gradients of weights
    int rows,cols;
    for(i=0;i<l.channels_in*l.channels_out;i++){
        rows = i/l.channels_out; //index of x
        cols = i%l.channels_out; //index of y
        l.weights[i] = l.weights[i] - l.lr*l.output.grad[cols]*l.input.data[rows];
    }

    // printf("layer weights data after update is: \n");
    // for(int i=0;i<l.channels_in*l.channels_out;i++){
    //     printf("%15.7f ",l.weights[i]);
    // }
    // printf("\n");

    //update gradients of bias
    for(i=0;i<l.channels_out;i++){
        l.bias[i] = l.bias[i] - l.lr*l.output.grad[i];
    }
}

//set gradients of input of this layer to be zero, in order for next episodic training
void zero_grad_fc_layer(fc_layer l){
    // for(int i=0;i<l.input.size;i++){
    //     l.input.grad[i] = 0.0;
    // }
    // //reset output data
    // for(int i=0;i<l.output.size;i++){
    //     l.output.data[i] = 0.0;
    // }
    //reset layer output data and gradients
    for(int i=0;i<l.output.size;i++){
        l.output.data[i] = 0.0;
        l.output.grad[i] = 0.0;
    }
}

