//
// Created by Axel Friberg on 2025-12-09.
//

#ifndef CNN_DENSE_H
#define CNN_DENSE_H
#include "activation.h"


typedef struct {
    int in_size; //4 0-3
    int out_size; //4 4-7

    double **weights; // 8 8-15
    double *biases; // 8 16-23

    double **dweights;
    double *dbiases;

    double **vweights;
    double *vbiases;
} DenseLayer;

DenseLayer* init_dense(int in_size, int out_size);
void dense_forw(DenseLayer *layer, double *dense_out, double *input, ActivationFn activation);

void update_dense(DenseLayer *layer, double learning_rate, double beta);
void zero_dense_gradients(DenseLayer *layer);
void free_dense(DenseLayer *layer);

#endif //CNN_DENSE_H