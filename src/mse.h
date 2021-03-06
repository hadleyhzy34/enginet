#ifndef MSE_H
#define MSE_H
#include <stdio.h>
#include <stdbool.h>
#include "enginet.h"

float mean_square_error(const unsigned int batch_size, tensor *output, tensor *label);
void zero_grad_mse(tensor output);

#endif
