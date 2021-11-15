#include <stdio.h>
#include <stdbool.h>
#include "enginet.h"

#define LR 0.01
#define EPOCHS 20

int main(){
    //matrix addition test
    float a[] = {1,2,3,4,5};
    float *ptr_a = a;

    float b[] = {2,3,4,5,6};
    float *ptr_b = b;

    float *a_b = gema(a,b,5);
    for(int i=0;i<5;i++){
        printf("%f ",a_b[i]);
    }
    
    /*
    float *ptr_c = (float*)calloc(100000,sizeof(float));
    for(int i=0;i<100000;i++){
        ptr_c[i] = i;
    }
    float *ptr_c_res = gema(ptr_c,ptr_c,100000);
    for(int i=0;i<100000;i++){
        printf("%f ",ptr_c_res[i]);
    }
    */
    
    /*---test for transpose---*/
    float d[] = {1,2,3,4,5,6};
    float *ptr_d = d;

    int d_shape[] = {2,3};
    int *d_shape_ptr = d_shape;

    //transpose(ptr_d, 2, 3);
    //print_matrix(ptr_d, 3, 2);

    transpose_gpu(ptr_d, 2, 3);    
    print_matrix(ptr_d, 3, 2);


    /*---test for matrix summation---*/
    size_t size = 10000;
    float *e = (float*)calloc(size,sizeof(float));
    for(int i=0;i<size;i++){
        e[i] = i+1;
    }
    float sum = gems(e, size);
    printf("\nmatrix addition result is: %f\n",sum);

    /*----test for matrix multiplication---*/
    float f[] = {1,2,3,4,5,6,7,8,9};
    float *ptr_f = f;

    float one[] = {1,0,0,0,1,0,0,0,1};
    float *ptr_one = one;

    float *res_f = (float*)calloc(9, sizeof(float));
    gemm_nt(3,3,3,1,ptr_f,3,ptr_one,3,res_f,3);
    printf("\nmatrix multiplication result is: \n");
    print_matrix(res_f,3,3);

    float two[] = {2,0,0,0,2,0,0,0,2};
    float *ptr_two = two;

    gemm_nt(3,3,3,1,ptr_f,3,ptr_two,3,res_f,3);
    printf("\nmatrix multiplication result is: \n");
    print_matrix(res_f,3,3);

    float h[] = {1,2,3,4,5,6,7,8,9,10,11,12};
    float *ptr_h = h;

    float *res_h = (float*)calloc(12, sizeof(float));
    gemm_nt(4,3,3,1,ptr_h,3,ptr_one,3,res_h,3);
    printf("\nmatrix multiplication result is: \n");
    print_matrix(res_h,4,3);

    gemm_nt(4,3,3,1,ptr_h,3,ptr_two,3,res_h,3);
    printf("\nmatrix multiplication result is: \n");
    print_matrix(res_h,4,3);

    /*------test for gemm-----*/
    printf("gemm test: \n");
    float *res_i = (float*)calloc(12,sizeof(float));
    gemm(0,1,4,3,3,1,ptr_h,3,ptr_two,3,0,res_i,3);
    print_matrix(res_i,4,3);

    return 0;
}
