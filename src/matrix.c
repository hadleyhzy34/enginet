#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matrix.h"

void print_matrix(matrix m)
{
    int i, j;
    printf("%d X %d Matrix:\n",m.rows, m.cols);
    printf(" __");
    for(j = 0; j < 16*m.cols-1; ++j) printf(" ");
    printf("__ \n");

    printf("|  ");
    for(j = 0; j < 16*m.cols-1; ++j) printf(" ");
    printf("  |\n");

    for(i = 0; i < m.rows; ++i){
        printf("|  ");
        for(j = 0; j < m.cols; ++j){
            printf("%15.7f ", m.vals[i][j]);
        }
        printf(" |\n");
    }
    printf("|__");
    for(j = 0; j < 16*m.cols-1; ++j) printf(" ");
    printf("__|\n");
}


matrix copy_matrix(matrix m)
{
    matrix c = {0};
    c.rows = m.rows;
    c.cols = m.cols;
    c.vals = calloc(c.rows, sizeof(float *));
    int i;
    for(i = 0; i < c.rows; ++i){
        c.vals[i] = calloc(c.cols, sizeof(float));
        for(int j = 0; j < c.cols; ++j){
            c.vals[i][j] = m.vals[i][j];
        }
    }
    return c;
}


matrix resize_matrix(matrix m, int rows, int cols){
    float **pointer = calloc(rows, sizeof(float*));
    int c_count = 0;
    int r_count = 0;
    for(int i=0;i<rows;i++){
       pointer[i] = calloc(cols,sizeof(float));
       for(int j=0;j<cols;j++){
           if(c_count<m.cols){
               pointer[i][j] = m.vals[r_count][c_count];
               c_count++;
           }else if(r_count+1<m.rows){
               r_count++;
               pointer[i][j] = m.vals[r_count][0];
               c_count=1;
           }
       }
    }
    free(m.vals);
    m.rows = rows;
    m.cols = cols;
    m.vals = pointer;
   return m;
}

/*
matrix resize_matrix(matrix m, int size)
{
    int i;
    if (m.rows == size) return m;
    if (m.rows < size) {
        m.vals = realloc(m.vals, size*sizeof(float*));
        for (i = m.rows; i < size; ++i) {
            m.vals[i] = calloc(m.cols, sizeof(float));
        }
    } else if (m.rows > size) {
        for (i = size; i < m.rows; ++i) {
            free(m.vals[i]);
        }
        m.vals = realloc(m.vals, size*sizeof(float*));
    }
    m.rows = size;
    return m;
}*/ 

//matrix addition
matrix mat_add(matrix a, matrix b){
    /*check dimension of matrix if it's correct*/
    if((a.cols != b.cols)||(a.rows != b.rows)){
        perror("dimension not compatible for matrix addition");
        exit(EXIT_FAILURE);
    }
    int rows = a.rows;
    int cols = a.cols;
    float **pointer = (float **)malloc(rows*sizeof(float*));

    for(int i=0;i<rows;i++){
        pointer[i] = (float*)malloc(4*sizeof(float));
        for(int j=0;j<cols;j++){
            pointer[i][j] = a.vals[i][j] + b.vals[i][j];
        }
    }
    matrix m ={rows,cols,pointer};
    return m;
}

//matrix multiplication
matrix mat_mul(matrix a, matrix b){
    /*check dimension of matrix if it's correct*/
    if(a.cols != b.rows){
        perror("dimension not compatible for matrix multiplication");
        exit(EXIT_FAILURE);
    }
    int rows = a.rows;
    int cols = b.cols;
    float **pointer = (float **)malloc(rows*sizeof(float*));

    for(int i=0;i<rows;i++){
        pointer[i] = (float*)malloc(4*sizeof(float));
        for(int j=0;j<cols;j++){
            pointer[i][j] = 0;
            for(int k=0;k<a.cols;k++){
                pointer[i][j] += a.vals[i][k] * b.vals[k][j];
            }
        }
    }
    matrix m ={rows,cols,pointer};
    return m;
}

//matrix substraction a-b
matrix mat_sub(matrix a, matrix b){
    /*check dimension of matrix if it's correct*/
    if((a.cols != b.cols)||(a.rows != b.rows)){
        perror("dimension not compatible for matrix addition");
        exit(EXIT_FAILURE);
    }
    int rows = a.rows;
    int cols = a.cols;
    float **pointer = (float **)malloc(rows*sizeof(float*));

    for(int i=0;i<rows;i++){
        pointer[i] = (float*)malloc(4*sizeof(float));
        for(int j=0;j<cols;j++){
            pointer[i][j] = a.vals[i][j] - b.vals[i][j];
        }
    }
    matrix m ={rows,cols,pointer};
    return m;
}

//matrix scalar, scalar * a
matrix mat_scal(matrix a, float scalar){
    int rows = a.rows;
    int cols = a.cols;
    float **pointer = (float **)malloc(rows*sizeof(float*));

    for(int i=0;i<rows;i++){
        pointer[i] = (float*)malloc(4*sizeof(float));
        for(int j=0;j<cols;j++){
            pointer[i][j] = a.vals[i][j] * scalar;
        }
    }
    matrix m ={rows,cols,pointer};
    return m;
}


