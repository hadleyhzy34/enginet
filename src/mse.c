#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matrix.h"
#include "tensor.h"

float mean_square_error(tensor output, tensor label){
    float loss=0.0;
    for(int i=0;i<output.size;i++){
        loss += (output.data[i] - label.data[i])*(output.data[i] - label.data[i]);
        // output.grad[i] += (1/(float)(output.size))*2*(output.data[i] - label.data[i]);
        output.grad[i] += 2*(output.data[i] - label.data[i]);
    }

    return loss;
}

void zero_grad_mse(tensor output){
    for(int i=0;i<output.size;i++){
        output.grad[i] = 0.0;  //reset gradients of tensor
        output.data[i] = 0.0;  //reset values of tensor
    }
}

