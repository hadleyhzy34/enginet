#include <stdio.h>
#include <stdbool.h>
#include "enginet.h"

#define LR 0.01
#define EPOCHS 20

int main(){
    printf("this is just a test\n");
    size_t dim = 4;
    size_t size = 100;
    float *t1 = (float *)calloc(size,sizeof(float));
    float *t2 = (float *)calloc(size,sizeof(float));
    
    int arr[] = {2,2,5,5};
    int *shape = arr;
    
    //test tensor_zeros(int *shape, size_t dim)
    tensor zero = tensor_zeros(shape, dim);
    print_tensor(zero); 

    //test tensor_ones(int *shape, size_t dim)
    tensor one = tensor_ones(shape, dim);
    print_tensor(one);
    printf("\n");

    //test tensor reshape func
    int dim1 = 3;
    int shape_1[] = {10,2,5};
    int *ptr_shape1 = shape_1;

    tensor *t_ptr = &one;
    reshape_tensor(t_ptr,ptr_shape1,dim1);
    print_tensor(one);
}
