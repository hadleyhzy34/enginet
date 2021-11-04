#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H
#include <stdio.h>
#include <stdbool.h>
#include "enginet.h"
#include "math.h"


ACTIVATION get_activation(char *s);
float activate(float x, ACTIVATION a);
void activate_array(float *x, const int n, const ACTIVATION a);
void activate_tensor(tensor input, tensor output, ACTIVATION a);

float activate_gradient(float x, ACTIVATION a);
void activate_gradient_array(const float *x, const int n, const ACTIVATION a, float *delta);

//activate function
static inline float relu_activate(float x){return x*(x>0);}
static inline float tanh_activate(float x){return (exp(2*x)-1)/(exp(2*x)+1);}

//activate function gradient
static inline float relu_gradient(float x){return (x>0);}
static inline float tanh_gradient(float x){return 1-x*x;}
#endif