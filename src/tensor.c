#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matrix.h"
#include "tensor.h"

void print_tensor(tensor t)
{
    int i, j;
    printf("Tensor size is: %d, %d\n",t.m.rows, t.m.cols);
    printf(" __");
    for(j = 0; j < 16*t.m.rows-1; ++j) printf(" ");
    printf("__ \n");

    printf("|  ");
    for(j = 0; j < 16*t.m.cols-1; ++j) printf(" ");
    printf("  |\n");

    for(i = 0; i < t.m.rows; ++i){
        printf("|  ");
        for(j = 0; j < t.m.cols; ++j){
            printf("%15.7f ", t.m.vals[i][j]);
        }
        printf(" |\n");
    }
    printf("|__");
    for(j = 0; j < 16*t.m.cols-1; ++j) printf(" ");
    printf("__|\n");
}

