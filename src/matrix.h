#ifndef MATRIX_H
#define MATRIX_H
#include "enginet.h"

matrix copy_matrix(matrix m);
void print_matrix(matrix m);

matrix resize_matrix(matrix m, int rows, int cols);

float *pop_column(matrix *m, int c);

#endif


