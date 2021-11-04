#ifndef BN_LAYER_H
#define BN_LAYER_H
#include <stdio.h>
#include <stdbool.h>
#include "enginet.h"

bn_layer ac_layer_initialization(const unsigned int batch_size, tensor *input, ACTIVATION a);
void forward_ac_layer(ac_layer l);
void backward_ac_layer(ac_layer l);
void zero_grad_ac_layer(ac_layer l);
#endif