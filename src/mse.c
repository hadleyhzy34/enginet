#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matrix.h"
#include "tensor.h"

float mean_square_error(const unsigned int batch_size, tensor *output, tensor *label){
    float loss=0.0;
    for(int k = 0; k < batch_size; k++){
        for(int i = 0; i < output[i].size; i++){
            loss += (output[k].data[i] - label[k].data[i]) * (output[k].data[i] - label[k].data[i]);
            output[k].grad[i] += 2*(output[k].data[i] - label[k].data[i]);
        }
    }
    return loss;
}

void zero_grad_mse(tensor output){
    for(int i=0;i<output.size;i++){
        output.grad[i] = 0.0;  //reset gradients of tensor
        output.data[i] = 0.0;  //reset values of tensor
    }
}

