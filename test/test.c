#include <stdio.h>
#include <stdbool.h>

/* 2d matrix */
typedef struct{
    int rows,cols;
    float **vals;
} matrix;

/*tensor*/
typedef struct{
    float grad;
    bool requires_grad;
    bool is_leaf;
    matrix m;
} tensor;

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
    matrix m ={3,4,pointer};

//     /*2d tensor*/
// typedef struct{
//     float grad;
//     bool is_leaf;
//     float ** vals;
//     int rows,cols;
//     bool requires_grad;
// } tensor;

    tensor tensor_test = {0.0, false, false, m};

    // m = resize_matrix(m,2,6);
    // print_matrix(m);

    // printf("current tensor matrix is: \n");
    // print_tensor(tensor_test);
    printf("current tensor size is: %d, %d\n", tensor_test.m.rows, tensor_test.m.cols);
}
