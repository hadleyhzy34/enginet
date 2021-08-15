#ifndef FC_LAYER_H
#define FC_LAYER_H
#include <stdio.h>
#include <stdbool.h>
#include "enginet.h"


fc_layer fc_layer_initialization(tensor input, int channels_in, int channels_out);
void forward_fc_layer(fc_layer l);
void backward_fc_layer(fc_layer l);
void zero_grad_fc_layer(fc_layer l);
#endif