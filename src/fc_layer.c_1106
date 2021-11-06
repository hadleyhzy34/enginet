#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matrix.h"
#include "tensor.h"
#include "fc_layer.h"

/*Generate new struct full connection layer
 *batch_size
 *tensor *input: array of tensor data with batch_size as length
 *in_channels: input dimension
 *out_channels: output dimension
 *return: struct fc_layer
*/
fc_layer fc_layer_initialization(const unsigned int batch_size, tensor *input, int in_channels, int out_channels){
    int i;
    //check each sample if dimension matched
    for(i = 0; i < batch_size; i++){
        if(input[i].size != in_channels){
            // printf("current input size is: %d, in_channels: %d, out_channels: %d\n", input[i].size, in_channels, out_channels);
            perror("Dimension of input tensor not compatible for matrix addition");
            exit(EXIT_FAILURE);
        }
    }
    float *weights = (float *)malloc(in_channels * out_channels * sizeof(float));   /*weight initialization*/

    for(i = 0; i < in_channels * out_channels; i++){
        weights[i] = ((float)rand()/RAND_MAX * 2) - 1;  /*size:in_channels * out_channels*/
    }
    float *bias = (float*)calloc(out_channels,sizeof(float)); /*bias initialization, size:out_channels*/
    tensor *output = (tensor*)malloc(batch_size*sizeof(tensor)); /*output batch memory space created*/
    for(i = 0; i < batch_size; i++){
        output[i] = tensor_initialization(out_channels, true);
    }
    fc_layer l = {input, output, batch_size, in_channels, out_channels, weights, bias}; /*fc_layer created*/
    return l;
}


void forward_fc_layer(fc_layer l){
    int i,j,k;
    for(k = 0; k < l.batch_size; k++){                  /*loop through each sample inside batch*/
        for(j = 0; j < l.out_channels; j++){            /*loop through each output dimension*/
            l.output[k].data[j] += l.bias[j];
            // printf("current batch: %d, l.output[k].data[j] is: %f", k, l.output[k].data[j]);
            for(i = 0; i < l.in_channels; i++){         /*loop through each input dimension*/
                l.output[k].data[j] += l.input[k].data[i] * l.weights[i*l.out_channels+j];
            }
        }
    }
    // for(j=0;j<l.channels_out;j++){
    //     l.output.data[j] += l.bias[j];
    //     for(i=0;i<l.channels_in;i++){
    //         l.output.data[j] += l.input.data[i] * l.weights[i*l.channels_out+j];
    //         // printf("i: %d, j: %d, l.input: %f, l.weights: %f, l.output: %f\n", i,j,l.input.data[i],l.weights[i*l.channels_out+j],l.output.data[j]);
    //     }
    // }
    // printf("tensor gradients for output is: \n");
    // print_tensor(l.output);
    // printf("\n");
}

void backward_fc_layer(fc_layer l){
    /* gradient of inputs
     *alpha(loss)/alpha(input) = output_gradients * alpha(output)/alpha(input)
    */
    int i,j,k;
    for(k = 0; k < l.batch_size; k++){
        for(i = 0; i < l.in_channels;i++){
            for(j = 0; j < l.out_channels;j++){
                l.input[k].grad[i] += l.output[k].grad[j] * l.weights[i*l.out_channels+j];
            }
        }
    }

    /*update gradients of weights, mini-batch gradients*/
    int rows,cols;
    for(i = 0; i < l.in_channels * l.out_channels; i++){
            rows = i / l.out_channels; /*index of input*/
            cols = i % l.out_channels; /*index of output*/
            float temp = 0.0;
            for(k = 0; k < l.batch_size; k++){
                temp += l.output[k].grad[cols] * l.input[k].data[rows];
            }
            l.weights[i] = l.weights[i] - LR * temp / (float)l.batch_size;
    }

    /*update gradients of bias, mini-batch gradients*/
    for(i=0;i<l.out_channels;i++){
        float temp = 0.0;
        for(k = 0; k < l.batch_size; k++){
            temp += l.output[k].grad[i];
        }
        l.bias[i] = l.bias[i] - LR * temp / (float)l.batch_size;
    }
}

/*reset gradients and data for fc layer*/
void zero_grad_fc_layer(fc_layer l){
    // for(int i=0;i<l.input.size;i++){
    //     l.input.grad[i] = 0.0;
    // }
    // //reset output data
    // for(int i=0;i<l.output.size;i++){
    //     l.output.data[i] = 0.0;
    // }
    //reset layer output data and gradients
    for(int k = 0; k < l.batch_size; k++){
        for(int i = 0; i < l.output[k].size; i++){
            l.output[k].data[i] = 0.0;
            l.output[k].grad[i] = 0.0;
        }
    }
}

