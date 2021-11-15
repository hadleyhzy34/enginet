#include <stdio.h>
#include <stdbool.h>
#include "enginet.h"

#define LR 0.01
#define EPOCHS 20

int main(){
    /*----test for matrix multiplication---*/
    float f[] = {1,2,3,4,5,6,7,8,9,10,11,12};
    float *ptr_f = f;

    //float one[] = {1,0,0,0,1,0,0,0,1};
    float one[] = {1,1,1,1,1,1,1,1,1};
    float *ptr_one = one;
    
    float *res_f = (float*)calloc(12, sizeof(float));
    
    //size_t m = 3;
    //size_t n = 4;
    //size_t k = 3;
    //gemm_gpu(0,0,m,n,k,1,ptr_one,m,ptr_f,k,0,res_f,m);
    mmul_gpu(4,3,3,ptr_f,ptr_one,res_f);
    print_matrix(res_f,4,3);

    float ones[] = {1,0,0,0,1,0,0,0,1};
    float *ptr_ones = ones;

    float *res_e = (float*)calloc(12,sizeof(float));
    mmul_gpu(4,3,3,ptr_f,ptr_ones,res_e);
    print_matrix(res_e,4,3);
    
    /*----------matrix multiplication---------*/
    float e[] = {4,5,6,7,8,9};
    float *ptr_e = e;

    float *res_g = (float*)calloc(6,sizeof(float));
    mmul_gpu(2,3,3,ptr_e,ptr_one,res_g);
    print_matrix(res_g,2,3);

    /*-----------array summation------------*/ 
    float sum = gems_gpu(ptr_f,12);
    printf("array sum is: %f\n",sum);
    
    float *g = (float*)malloc(10000*sizeof(float));
    for(int i=0;i<10000;i++){
        g[i] = i+1.f;
    }
    
    float sum_1 = gems_gpu(g,10000);
    printf("array sum is: %f\n",sum_1);


    /*--------------matrix addition test------------*/
    float nones[] = {1,1,1,1,1,1,1,1,1,1,1,1};
    float *ptr_nones = nones;

    float *res_h = (float*)calloc(12,sizeof(float));
    gema_gpu(ptr_f,ptr_nones,res_h,12);
    print_matrix(res_h,12,1);
    
    /*---test for transpose---*/                                                
    float d[] = {1,2,3,4,5,6};                                                  
    float *ptr_d = d;                                                           
                                                                                
    int d_shape[] = {2,3};                                                      
    int *d_shape_ptr = d_shape;                                                 
                                                                                
                                                                                 
    transpose_gpu(ptr_d, 2, 3);                                                 
    print_matrix(ptr_d, 3, 2);

    float i[] = {0,2,4,6,8,10,12,14,16,18,20,22};
    float *ptr_i = i;

    transpose_gpu(ptr_i, 3, 4);
    print_matrix(ptr_i, 4, 3);

    float h[] = {1,3,5,7,9,11,2,4,6,8,10,12};
    float *ptr_h = h;

    transpose_gpu(ptr_h, 2, 6);
    print_matrix(ptr_h, 6, 2);
    //gemm_gpu(1,1,4,3,3,1,ptr_f,4,ptr_one,3,0,res_f,4);

    /* 
    float *d_A, *d_B, *d_C;                                                         
    cudaMalloc(&d_A, sizeof(float)*9);                      
    cudaMalloc(&d_B, sizeof(float)*9);                      
    cudaMalloc(&d_C, sizeof(float)*9);                      
    cudaMemcpy(d_A,ptr_f,sizeof(float)*9,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,ptr_one,sizeof(float)*9,cudaMemcpyHostToDevice);
    cudaMemcpy(d_C,res_f,sizeof(float)*9,cudaMemcpyHostToDevice);

    gemm_gpu(0,0,3,3,3,1,d_A,3,d_B,3,0,d_C,3);
    cudaMemcpy(res_f,d_C,sizeof(float)*9,cudaMemcpyDeviceToHost);
    print_matrix(res_f,3,3);
    */

    /*
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
    */

    /*------test for gemm-----*/

    /*
    printf("gemm test: \n");
    float *res_i = (float*)calloc(12,sizeof(float));
    gemm(0,1,4,3,3,1,ptr_h,3,ptr_two,3,0,res_i,3);
    print_matrix(res_i,4,3);
    */

    return 0;
}
