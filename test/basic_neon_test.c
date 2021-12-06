#include <stdio.h>
#include <stdbool.h>
#include <time.h>
#include "enginet.h"

#define LR 0.01
#define EPOCHS 20

int main(){
    float a[] = {0,1,2,3,4,5};
    float b[] = {1,1,1,1,1,1};
    printf("gems arm output is: %.5f\n",gems_arm(&a,6));
    printf("gevdp arm output is: %.5f\n", gevdp_arm(&a,&b,6));
    
    size_t size = 103;
    float* c = (float*)calloc(size, sizeof(float));
    float* d = (float*)calloc(size, sizeof(float));

    for(int i=0;i<size;i++){
        c[i] = i+1.f;
        d[i] = 1.f;
    }

    float *output = (float*)malloc(size*sizeof(float));
    output = gema_arm(c,d,size);
    for(int i=0;i<size;i++){
        printf("%f ",output[i]);
    }
    
    printf("\n");

    printf("gemc output value is: \n");
    gemc_arm(c, size, 2.f);
    for(int i=0;i<size;i++){
        printf("%f ",c[i]);
    }
    printf("\n");

    printf("geac output value is: \n");
    geac_arm(c, size, 2.f);
    for(int i=0;i<size;i++){
        printf("%f ",c[i]);
    }
    printf("\n");

    float *res = (float*)malloc(6*sizeof(float));
    res = gema_arm(&a,&b,6);
    for(int i=0;i<6;i++){
        printf("current element is: %f", res[i]);
    }
    return 0;
}
