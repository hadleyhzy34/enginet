#include <stdio.h>
#include <stdbool.h>
#include "enginet.h"

#define LR 0.01
#define EPOCHS 50

int main(){
    unsigned int size = 10;
    float *v0 = (float *)calloc(size,sizeof(float));
    float *v1 = (float *)calloc(size,sizeof(float));
    float *v2 = (float *)calloc(size,sizeof(float));
    float *v3 = (float *)calloc(size,sizeof(float));

    for(int i=0;i<size;i++){
        v0[i] = i+1.0;
        v2[i] = 2*(i+1.0);
    }
    
    tensor t_input_1 = {v0, v1, size,  true};      //input tensor 1 value:{1,2,3} gradients{0,0,0}
    tensor t_label_2 = {v2, v3, size,  true};      //label tensor 2 label:{2,4,6} gradients{0,0,0}

    
    printf("current tensor input data is: \n");
    print_tensor(t_input_1);

    //layer initialization
    unsigned int hidden_size = 50;
    // fc_layer l1 = fc_layer_initialization(t_input_1, t_input_1.size, t_label_2.size);
    fc_layer l1 = fc_layer_initialization(t_input_1, t_input_1.size, hidden_size);
    ac_layer l2 = ac_layer_initialization(l1.output, RELU);
    fc_layer l3 = fc_layer_initialization(l2.output, l1.output.size, t_label_2.size);
    float loss = 0.0;
    //training process
    for(int i=0;i<EPOCHS;i++){
        printf("--------------new episode: -----------------------\n");
        //forward operation
        forward_fc_layer(l1);
        forward_ac_layer(l2);
        forward_fc_layer(l3);
        loss = mean_square_error(l3.output,t_label_2);
        printf("--------------current epochs %d, loss: %f-------------\n", i, loss);

        //backward operation
        backward_fc_layer(l3);
        backward_ac_layer(l2);
        backward_fc_layer(l1);

        // printf("tensor gradients for output is: \n");
        // // print_tensor(l1.input);
        // printf("\n");
        // print_tensor(l1.output);
        // printf("\n");
        // print_tensor(l1.input);
        // printf("\n");

        //zero grad
        // zero_grad_mse(l2.output); //reset l2.output data and gradients
        zero_grad_fc_layer(l3);   //reset l3 output data and gradients
        zero_grad_ac_layer(l2);   //reset l2 output data and gradients
        zero_grad_fc_layer(l1);   //reset l1 output data and gradients
        printf("-----------tensor gradients after zero gradients is: ------------\n");
        printf("---l1 input---\n");
        print_tensor(l1.input);
        printf("---l1 output---\n");
        print_tensor(l1.output);
        printf("---l2 input---\n");
        print_tensor(l2.input);
        printf("---l2 output---\n");
        print_tensor(l2.output);
        printf("---l3 input---\n");
        print_tensor(l3.input);
        printf("---l3 output---\n");
        print_tensor(l3.output);
        printf("\n");



        // printf("layer l1 weights data is: \n");
        // for(int i=0;i<l1.channels_in*l1.channels_out;i++){
        //     printf("%15.7f ",l1.weights[i]);
        // }
        // printf("\n");

        // printf("layer bias data is: \n");
        // for(int i=0;i<l1.channels_out;i++){
        //     printf("%15.7f ",l1.bias[i]);
        // }
        // printf("\n");

        // printf("layer l2 weights data is: \n");
        // for(int i=0;i<l2.channels_in*l2.channels_out;i++){
        //     printf("%15.7f ",l2.weights[i]);
        // }
        // printf("\n");

        // printf("layer bias data is: \n");
        // for(int i=0;i<l2.channels_out;i++){
        //     printf("%15.7f ",l2.bias[i]);
        // }
        // printf("\n");
    }
    printf("----evaluation----\n");
    forward_fc_layer(l1);
    forward_ac_layer(l2);
    forward_fc_layer(l3);
    printf("---output of fc layer is: ----\n");
    print_tensor(l3.output);
    // printf("----fc layer parameters are: ----\n");
    // for(int i=0;i<l1.channels_in*l1.channels_out;i++){
    //     printf("%15.7f ",l2.weights[i]);
    // }
    // printf("\n---fc layer bias updated value are:\n");
    // for(int j=0;j<l1.channels_out;j++){
    //     printf("%15.7f ", l2.bias[j]);
    // }
    // printf("\n");
}
