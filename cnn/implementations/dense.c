//
// Created by Axel Friberg on 2025-12-09.
//

#include "../headers/dense.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <tgmath.h>
#include "../headers/activation.h"


DenseLayer* init_dense(int in_size, int out_size) {

    DenseLayer *layer = malloc(sizeof(DenseLayer));
    assert(layer);

    layer->in_size = in_size;
    layer->out_size = out_size;

    layer->weights = malloc(sizeof(double*) * out_size);
    layer->biases = malloc(sizeof(double) * out_size);
    layer->dweights = malloc(sizeof(double*) * out_size);
    layer->dbiases = malloc(sizeof(double) * out_size);
    layer->vweights = malloc(sizeof(double*) * out_size);
    layer->vbiases = malloc(sizeof(double) * out_size);


    int fan_in = in_size;
    double limit = sqrt(6.0/fan_in); // he-uniform again

    for (int i = 0; i < out_size; i++) {
        layer->weights[i] = malloc(sizeof(double) * in_size);
        layer->dweights[i] = malloc(sizeof(double) * in_size);
        layer->vweights[i] = malloc(sizeof(double*) * in_size);
        assert(layer->weights[i]);
        for (int j = 0; j < in_size; j++) {
            layer->weights[i][j] = ((double)rand() / RAND_MAX) * 2.0 * limit - limit;
        }
        layer->biases[i] = 0.01;
        layer->dbiases[i] = 0.0;
        layer->vbiases[i] = 0.0;
    }

    return layer;
}

void dense_forw(DenseLayer *layer, double *dense_out, double *input, ActivationFn activation) {

    //double *output = malloc(sizeof(double) * layer->out_size);
    //assert(output);

    for (int i = 0; i < layer->out_size; i++) {
        double weighted_sum = 0.0;
        for (int j = 0; j < layer->in_size; j++) {
            weighted_sum += layer->weights[i][j] * input[j];
        }
        dense_out[i] = activation(weighted_sum + layer->biases[i]);
    }
}



void update_dense(DenseLayer *layer, double learning_rate, double beta) {

    for (int i = 0; i < layer->out_size; i++) {

        for (int j = 0; j < layer->in_size; j++) {

            // update to accomodate velocity
            layer->vweights[i][j] = beta * layer->vweights[i][j] + layer->dweights[i][j];

            layer->weights[i][j] -= learning_rate * layer->vweights[i][j];
        }

        layer->vbiases[i] =
            beta * layer->vbiases[i] + layer->dbiases[i];

        layer->biases[i] -= learning_rate * layer->vbiases[i];
    }
}

void zero_dense_gradients(DenseLayer *layer) {

    memset(layer->dbiases, 0, sizeof(double) * layer->out_size);

    for (int i = 0; i < layer->out_size; i++) {

        memset(layer->dweights[i], 0, sizeof(double) * layer->in_size);

    }
}




void free_dense(DenseLayer *layer) {
    if (!layer) return;

    for (int i = 0; i < layer->out_size; i++) {
        free(layer->weights[i]);
        free(layer->dweights[i]);
        free(layer->vweights[i]);
    }

    free(layer->weights);
    free(layer->dweights);
    free(layer->vweights);

    free(layer->biases);
    free(layer->dbiases);
    free(layer->vbiases);

    free(layer);
}

