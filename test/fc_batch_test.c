#include <stdio.h>
#include <stdbool.h>
#include "enginet.h"


int main(){
    const unsigned int size = 5;    /*input tensor data size*/

    /*array of tensor initialization*/
    tensor *t_input = (tensor*)malloc(BATCH_SIZE*sizeof(tensor));
    tensor *t_label = (tensor*)malloc(BATCH_SIZE*sizeof(tensor));

    int i,k;
    for(k = 0; k < BATCH_SIZE; k++){
        t_input[k] = tensor_initialization(size, true);
        t_label[k] = tensor_initialization(size, true);
        for(i = 0; i < size; i++){
            t_input[k].data[i] = i + k + 1.0;
            t_label[k].data[i] = 2.0 * (i + k + 1.0);
        }
        printf("---batch index: %d, current input data: \n", k);
        print_tensor(t_input[k]);
        printf("---current lable data: \n");
        print_tensor(t_label[k]);
        printf("\n");
    }

    //layer initialization
    unsigned int hidden_size = 20;
    // fc_layer l1 = fc_layer_initialization(t_input_1, t_input_1.size, t_label_2.size);
    fc_layer l1 = fc_layer_initialization(BATCH_SIZE, t_input, size, hidden_size);
    ac_layer l2 = ac_layer_initialization(BATCH_SIZE, l1.output, RELU);
    fc_layer l3 = fc_layer_initialization(BATCH_SIZE, l2.output, hidden_size, size);
    ac_layer l4 = ac_layer_initialization(BATCH_SIZE, l3.output, RELU);
    
    float loss = 0.0;
    //training process
    for(int i=0;i<EPOCHS;i++){
        printf("--------------new episode: -----------------------\n");
        //forward operation
        printf("forward op 1");
        forward_fc_layer(l1);
        printf("forward op 2");
        forward_ac_layer(l2);
        printf("forward op 3");
        forward_fc_layer(l3);
        printf("mse ops");
        forward_ac_layer(l4);
        loss = mean_square_error(BATCH_SIZE, l3.output, t_label);
        printf("--------------current epochs %d, loss: %f-------------\n", i, loss);

        //backward operation
        backward_fc_layer(l3);
        backward_ac_layer(l2);
        backward_fc_layer(l1);
        backward_ac_layer(l4);

        // printf("tensor gradients for output is: \n");
        // // print_tensor(l1.input);
        // printf("\n");
        // print_tensor(l1.output);
        // printf("\n");
        // print_tensor(l1.input);
        // printf("\n");

        //zero grad
        // zero_grad_mse(l2.output); //reset l2.output data and gradients
        zero_grad_ac_layer(l4);
        zero_grad_fc_layer(l3);   //reset l3 output data and gradients
        zero_grad_ac_layer(l2);   //reset l2 output data and gradients
        zero_grad_fc_layer(l1);   //reset l1 output data and gradients
        // printf("-----------tensor gradients after zero gradients is: ------------\n");
        // unsigned int k;
        // for(k = 0; k < BATCH_SIZE; k++){
        //     printf("---l1 input---\n");
        //     print_tensor(l1.input[k]);
        //     printf("---l1 output---\n");
        //     print_tensor(l1.output[k]);
        //     printf("---l2 input---\n");
        //     print_tensor(l2.input[k]);
        //     printf("---l2 output---\n");
        //     print_tensor(l2.output[k]);
        //     printf("---l3 input---\n");
        //     print_tensor(l3.input[k]);
        //     printf("---l3 output---\n");
        //     print_tensor(l3.output[k]);
        //     printf("\n");
        // }



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
    forward_ac_layer(l4);
    printf("---output of fc layer is: ----\n");
    print_tensor(t_label[0]);
    print_tensor(l4.output[0]);
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
