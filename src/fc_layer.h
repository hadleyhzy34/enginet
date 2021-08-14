#ifndef FC_LAYER_H
#define FC_LAYER_H
#include <stdio.h>
#include <stdbool.h>
#include "enginet.h"

void forward_fc_layer(fc_layer l);

void backward_fc_layer(fc_layer l);

#endif