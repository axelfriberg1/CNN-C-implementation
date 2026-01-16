//
// Created by Axel Friberg on 2025-12-05.
//

#ifndef CNN_CONVLAYER_H
#define CNN_CONVLAYER_H
#include <assert.h>
#include <stdlib.h>
#include "../headers/activation.h"


typedef struct {
    int numFilters;
    int filterSize;

    double ***filters;
    double *biases;

    double ***dfilters;
    double *dbiases;

    double ***vfilters;
    double *vbiases;
} ConvLayer;


ConvLayer* init_convlayer(int numFilters, int filterSize);
void conv_forward(ConvLayer *layer, double ***feature_out, double** image, int in_h, int in_w, ActivationFn activation_fn);
void conv(
    double** image,
    double** output,
    double **filter,
    int filter_h,
    int filter_w,
    int in_h, int in_w,
    double bias,
    ActivationFn activation_fn
    );

void update_conv(ConvLayer *layer, double learning_rate, double beta);
void zero_conv_gradients(ConvLayer *layer);
void free_convlayer(ConvLayer *layer);

#endif //CNN_CONVLAYER_H