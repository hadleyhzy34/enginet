#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matrix.h"
#include "tensor.h"
#include "fc_layer.h"

// /*fc_layer*/
// typedef struct{
//     tensor input;  //input data
//     tensor output;  //output data
//     int input_size;
//     int output_size;
//     tensor weights;  //fc layer weights
//     tensor bias;  // fc layer bias
//     float lr;  //learning rate for updating fc layer
// } fc_layer;


void forward_fc_layer(fc_layer l){
    /*------------------------------------layer initialization*----------------------------------------------*/
    //weights and gradients initialization, size: l.channels_in * l.channels_out
    if(l.weights==NULL){
        float **weights_dat=(float **)malloc(l.channels_in*sizeof(float*));
        float **grad_w_dat=(float **)malloc(l.channels_in*sizeof(float*));
        for(int i=0;i<l.channels_in;i++){
            weights_dat[i] = (float*)malloc(l.channels_out*sizeof(float));
            grad_w_dat[i] = (float*)malloc(l.channels_out*sizeof(float));
            for(int j=0;j<l.channels_out;j++){
                weights_dat[i][j] = (float)(rand()%100)/float(100);
                grad_w_dat[i][j] = 0.0;
            }
        }
        matrix w_d = {l.channels_in,l.channels_out,weights_dat};
        matrix g_w_d = {l.channels_in,l.channels_out,grad_w_dat};
        l.weights={g_w_d, false, false, w_d};

        //debug
        print_tensor(l.weights);
    }

    //bias and gradients initialization, size: l.channels_out * 1
    if(l.bias==NULL){
        float **bias_dat=(float **)malloc(l.channels_out*sizeof(float*));
        float **grad_b_dat=(float **)malloc(l.channels_out*sizeof(float*));
        for(int i=0;i<l.channels_out;i++){
            bias_dat[i] = (float*)malloc(sizeof(float));
            bias_dat[i][0] = (float)(rand()%100)/float(100);

            grad_b_dat[i] = (float*)malloc(sizeof(float));  
            grad_b_dat[i][0] = 0.0;
        }
        matrix b_d = {l.channels_out,1,bias_dat};
        matrix g_b_d = {l.channels_out,1,grad_b_dat};
        l.weights={b_d, false, false, g_b_d};

        //debug
        print_tensor(l.bias);
    }
    //output tensor initialization
    if(l.output==NULL){
        float **output_dat=(float **)malloc(l.channels_out*sizeof(float*));
        float **grad_o_dat=(float **)malloc(l.channels_out*sizeof(float*));
        for(int i=0;i<l.channels_out;i++){
            output_dat[i] = (float*)malloc(sizeof(float));
            output_dat[i][0] = 0.0;

            grad_o_dat[i] = (float*)malloc(sizeof(float));  
            grad_o_dat[i][0] = 0.0;
        }
        matrix o_d = {l.channels_in,l.channels_out,weights_dat};
        matrix g_o_d = {l.channels_in,l.channels_out,grad_w_dat};
        l.weights={g_o_d, false, false, o_d};

        //debug
        print_tensor(l.output);
    }
    /*------------------------------------layer forward operation----------------------------------------------*/
    l.output.m = mat_add(mat_mul(l.input.m, l.weights.m), l.bias.m);
}

void backward_fc_layer(fc_layer l){
    //layer gradient update
    l.input.grad = mat_mul(l.output.grad, l.weights.m);
    l.weights.grad = mat_mul(l.output.grad, l.input.m);
    l.bias.grad = l.output.grad;

    //layer weights,bias gradient descent
    l.weights.m = mat_sub(l.weights.m, mat_scal(l.weights.grad, l.lr));
    l.bias.m = mat_sub(l.bias.m, mat_scal(l.bias.grad, l.lr));
}

