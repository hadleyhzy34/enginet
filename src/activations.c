#include "activations.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

ACTIVATION get_activation(char *s)
{
    if (strcmp(s, "relu")==0) return RELU;
    if (strcmp(s, "tanh")==0) return TANH;
    fprintf(stderr, "Couldn't find activation function %s, going with ReLU\n", s);
    return RELU;
}

float activate(float x, ACTIVATION a)
{
    switch(a){
        case RELU:
            return relu_activate(x);
        case TANH:
            return tanh_activate(x);
    }
    return 0;
}

void activate_array(float *x, const int n, const ACTIVATION a)
{
    int i;
    for(i = 0; i < n; ++i){
        x[i] = activate(x[i], a);
    }
}

void activate_tensor(tensor input, tensor output, ACTIVATION a){
    if(input.size != output.size){
        perror("dimension not compatible for matrix activation");
        exit(EXIT_FAILURE);
    }
    unsigned int i;
    for(i = 0; i < input.size; i++){
        output.data[i] = activate(input.data[i], a);
    }
}

float activate_gradient(float x, ACTIVATION a)
{
    switch(a){
        case RELU:
            return relu_gradient(x);
        case TANH:
            return tanh_gradient(x);
    }
    return 0;
}

void activate_gradient_array(const float *x, const int n, const ACTIVATION a, float *delta)
{
    int i;
    for(i = 0; i < n; ++i){
        delta[i] *= activate_gradient(x[i], a);
    }
} 