//
// Created by Axel Friberg on 2025-12-19.
//

#ifndef CNN_BACKPROP_H
#define CNN_BACKPROP_H
#include "convlayer.h"
#include "dense.h"
#include "pool.h"


double cross_entropy(double *preds, int true_label, int num_classes);
double softmax_cross_entropy_back(double *preds, int true_label, double *dlogits, int num_classes);
void dense_back_no_activation(
    DenseLayer *layer,
    const double *input,
    const double *dlogits,
    double *dinput
);

void dense_back_relu(
    DenseLayer *layer,
    double *layer_out,
    const double *input,
    const double *dout,
    double *dinput);


void pool_back(PoolingLayer *layer, double ***feature_maps, double*** pooled_images, double ***dout, double ***dinput);
void max_pool_back(double **feature_map, double **pooled_images, double **dout, double **dinput);
void conv_back(
    ConvLayer *layer,
    double **input_image,        // [28][28]
    double ***feature_maps,      // [6][26][26] (after ReLU)
    double ***dout,              // [6][26][26] (from pooling back)
    double ***dconv              // [6][28][28] (output buffer)
);
#endif //CNN_BACKPROP_H