#ifndef ENGINET_API
#define ENGINET_API
#include <stdlib.h>

typedef struct{
    int rows,cols;
    float **vals;
} matrix;

void print_matrix(matrix m);
matrix resize_matrix(matrix m, int rows, int cols);
void free_matrix(matrix m);
void matrix_to_csv(matrix m);
void scale_matrix(matrix m, float scale);
#endif
