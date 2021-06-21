#ifndef ENGINET_API
#define ENGINET_API
#include <stdlib.h>

typedef struct{
    int rows,cols;
    float **vals;
} matrix;

void print_matrix(matrix m);

#endif
