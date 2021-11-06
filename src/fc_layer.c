#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matrix.h"
#include "tensor.h"
#include "fc_layer.h"
#include "core/gemm.h"

/*Generate new struct full connection layer
 *batch_size
 *tensor *input: array of tensor data with batch_size as length
 *in_channels: input dimension
 *out_channels: output dimension
 *return: struct fc_layer
*/

fc_layer fc(size_t in_channels, size_t out_channels){
    fc_layer l;
    l.weights = (float* )calloc(in_channels*out_channels,sizeof(float));
    l.bias = (float* )calloc(out_channels,sizeof(float));
    return l;
}

void forward_fc(tensor *input, fc_layer l, struct graph_node parent){
    struct graph_node cur;
    cur.parent = parent;
    l.output = tensor_zeros(input.shape,2);
    for(int i=0;i<input.shape[0];i++){
        l.output[i*l.out_channels] = gemm(0,1,input.shape[1],)
    }
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

