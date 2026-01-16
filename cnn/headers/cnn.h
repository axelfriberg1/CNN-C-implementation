//
// Created by Axel Friberg on 2025-12-31.
//

#ifndef CNN_CNN_H
#define CNN_CNN_H
#include <stdio.h>

#include "convlayer.h"
#include "dense.h"
#include "pool.h"

typedef struct {
    double learning_rate;
    double momentum;
    int batch_size;
} HyperParams;

typedef struct {
    size_t img_h, img_w;
    HyperParams *hyper_params;
    ConvLayer *conv1;
    PoolingLayer *pool1;
    DenseLayer *dense1, *dense2, *dense3;

} Cnn;

typedef struct {

    // forward buffers
    double ***conv1_buffer;
    double ***pool1_buffer;
    double *flat_buffer;
    double *dense1_buffer;
    double *dense2_buffer;
    double *dense3_buffer;
    double *probs_buffer;

    double *grad_logits;
    double *grad_dense2;
    double *grad_dense1;
    double *grad_flat;
    double ***grad_pool;
    double ***grad_conv;
    double ***grad_input;
} CnnMemory;

Cnn *init_cnn(
    ConvLayer *conv1,
    PoolingLayer *pool1,
    DenseLayer *dense1,
    DenseLayer *dense2,
    DenseLayer *dense3,
    size_t img_h,
    size_t img_w);

CnnMemory *init_memory(Cnn *cnn);

void cnn_zero_forward_buffers(Cnn *cnn, CnnMemory *memory);
void cnn_zero_backward_buffers(Cnn *cnn, CnnMemory *memory);

void free_memory(CnnMemory *memory, Cnn *cnn);

#endif //CNN_CNN_H