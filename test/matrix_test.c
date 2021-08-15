#include <stdio.h>
#include <stdbool.h>
#include "enginet.h"

#define LR 0.01
#define EPOCHS 10

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

// //tensor struct
// typedef struct{
//     float *data; //tensor data
//     float *grad; //gradients, alpha(loss)/ahpha(x)
//     int size;
//     bool requires_grad;
// }tensor;

    float *v1 = (float *)calloc(12,sizeof(float));
    float *v2 = (float *)calloc(12,sizeof(float));
    float *v3 = (float *)calloc(24,sizeof(float));
    float *v4 = (float *)calloc(24,sizeof(float));
    float *v5 = (float *)calloc(24,sizeof(float));
    for(int i=0;i<24;i++){
        v3[i] = (float)i;
        v5[i] = 1.0;
    }
    
    tensor t1 = {v1, v2, 12, true};
    tensor t2 = {v3, v4, 24, true};
    tensor t_label = {v5, v2, 24, true};

    // m = resize_matrix(m,2,6);
    // print_matrix(m);

    // printf("current tensor matrix is: \n");
    print_tensor(t1);
    print_tensor(t2);

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

    //layer initialization
    fc_layer l1 = fc_layer_initialization(t2, 24, 24);
    printf("layer output data is: \n");
    print_tensor(l1.output);

    printf("layer weights data is: \n");
    for(int i=0;i<l1.channels_in*l1.channels_out;i++){
        printf("%15.7f ",l1.weights[i]);
    }
    printf("\n");

    // //forward fc layer
    // forward_fc_layer(l1);
    // printf("layer output data after forward operation is: \n");
    // print_tensor(l1.output);


    // float loss = 0.0;
    // loss += mean_square_error(l1.output,t_label);
    // printf("current loss is: %15.7f\n", loss);
    // printf("layer output data after loss calculation is: \n");
    // print_tensor(l1.output);

    // //backward fc layer
    // backward_fc_layer(l1);
    // printf("layer output data after backward operation is: \n");
    // print_tensor(l1.input);

    // printf("layer weights data is: \n");
    // for(int i=0;i<l1.channels_in*l1.channels_out;i++){
    //     printf("%15.7f ",l1.weights[i]);
    // }
    // printf("\n");

    // printf("layer bias data is: \n");
    // for(int i=0;i<l1.channels_out;i++){
    //     printf("%15.7f ",l1.bias[i]);
    // }
    // printf("\n");

    float loss = 0.0;
    //training process
    for(int i=0;i<EPOCHS;i++){
        //forward operation
        forward_fc_layer(l1);
        loss = mean_square_error(l1.output,t_label);
        printf("current epochs %d, loss: %f\n", i, loss);

        //backward operation
        backward_fc_layer(l1);

        //zero grad
        zero_grad_mse(l1.output);
        zero_grad_fc_layer(l1);
    }
}
