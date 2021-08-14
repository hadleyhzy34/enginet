#include <stdio.h>
#include <stdbool.h>
#include "enginet.h"

#define LR 0.01

int main(){
    printf("this is just a test\n");
    float t[3][4] = {
                    {1.0,2.0,3.0,4.0},
                    {5.0,6.0,7.0,8.0},
                    {9.0,10.0,11.0,12.0}
    };
    float **pointer = (float **)malloc(3*sizeof(float*));
    for(int r=0;r<3;r++){
        pointer[r] = (float*)malloc(4*sizeof(float));
        for(int c=0;c<4;c++){
            pointer[r][c] = t[r][c];
        }
    }

    float **pointer1 = (float **)malloc(3*sizeof(float*));
    for(int r=0;r<3;r++){
        pointer1[r] = (float*)malloc(4*sizeof(float));
        for(int c=0;c<4;c++){
            pointer1[r][c] = t[r][c]*2;
        }
    }

    float diagnol_mat[3][3] = {
                    {1.0,0.0,0.0},
                    {0.0,1.0,0.0},
                    {0.0,0.0,1.0}
    };
    float **pointer2 = (float **)malloc(3*sizeof(float*));
    for(int r=0;r<3;r++){
        pointer2[r] = (float*)malloc(4*sizeof(float));
        for(int c=0;c<3;c++){
            pointer2[r][c] = diagnol_mat[r][c];
        }
    }

    matrix m1 = {3, 4, pointer};
    matrix m2 = {3, 4, pointer1};
    matrix m5 = {3, 3, pointer2}; //diagnol matrix
    matrix m3 = mat_add(m1, m2);
    matrix m4 = mat_mul(m5, m2);
    matrix m6 = mat_sub(m2,m1);
    matrix m7 = mat_scal(m2, 2.0);

    print_matrix(m6);
    print_matrix(m7);

// /* 2d matrix */
// typedef struct{
//     int rows,cols;
//     float **vals;
// } matrix;

// /*tensor*/
// typedef struct{
//     matrix grad;
//     bool requires_grad;
//     bool is_leaf;
//     matrix m;
// } tensor;

    tensor test = {{}, false, false, m1};
    tensor tensor_test = {{}, false, false, m4};

    // m = resize_matrix(m,2,6);
    // print_matrix(m);

    // printf("current tensor matrix is: \n");
    print_tensor(tensor_test);

// /*fc_layer*/
// typedef struct{
//     tensor input;  //input data
//     tensor output;  //output data
//     // int input_size;
//     // int output_size;
//     tensor weights;  //fc layer weights
//     tensor bias;  // fc layer bias
//     float lr;  //learning rate for updating fc layer
// } fc_layer;

    //tensor input
    tensor tensor_in = {{},false, false, m1};
    //layer initialization
    fc_layer l1 = {tensor_in, {}, {}, {}, LR};

    forward_fc_layer(l1);
}
