#include <stdio.h>
#include <stdbool.h>
#include "enginet.h"

#define LR 0.01
#define EPOCHS 20

int main(){
    printf("this is just a test\n");

    //simple matrix multiplication to verify gemm
    float a[] = {1,2,3,4,5,6};
    float *A = a;

    float b[] = {1,2,1,2,1,2};
    float *B = b;

    float c[] = {0,0,0,0};
    float *C = c;

    gemm(0,1,2,2,3,1,A,3,B,2,0,C,2);

    for(int i=0;i<4;i++){
        printf("%f ",C[i]);
    }
}
