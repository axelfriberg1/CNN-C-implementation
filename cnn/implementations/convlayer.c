//
// Created by Axel Friberg on 2025-12-05.
//


#include <stdlib.h>
#include "../headers/convlayer.h"

#include <string.h>

#include "../headers/activation.h"
#include <tgmath.h>


ConvLayer* init_convlayer(int numFilters, int filterSize) {
    ConvLayer *layer = malloc(sizeof(ConvLayer));

    assert(layer != NULL);

    layer->numFilters = numFilters;
    layer->filterSize = filterSize;
    layer->filters = malloc(numFilters * sizeof(double**));
    layer->biases = malloc(numFilters * sizeof(double));
    layer->dfilters = malloc(numFilters * sizeof(double**));
    layer->dbiases = calloc(numFilters , sizeof(double));
    layer->vfilters = malloc(numFilters * sizeof(double**));
    layer->vbiases = calloc(numFilters , sizeof(double));

    //
    // https://keras.io/api/layers/initializers/
    double fan_in = filterSize * filterSize;  // we use only 1 input channel
    double limit = sqrt(6.0 / fan_in);

    for (int i = 0; i < numFilters; i++) {
        layer->filters[i] = malloc(sizeof(double*) * filterSize);
        layer->dfilters[i] = malloc(sizeof(double*) * filterSize);
        layer->vfilters[i] = malloc(sizeof(double*) * filterSize);
        layer->biases[i] = 0;
        for (int j = 0; j < filterSize; j++) {
            layer->filters[i][j] = malloc(sizeof(double) * filterSize);
            layer->dfilters[i][j] = calloc(filterSize, sizeof(double));
            layer->vfilters[i][j] = calloc(filterSize, sizeof(double));
            for (int k = 0; k < filterSize; k++) {
                // he uniform initialization
                // https://apxml.com/courses/how-to-build-a-large-language-model/chapter-12-initialization-techniques-deep-networks/kaiming-he-initialization
                double w = ( ((double)rand() / RAND_MAX) * 2.0 * limit ) - limit;
                layer->filters[i][j][k] = w;
            }
        }
    }

    return layer;
}

/*
 * Performs the actual convolutions.
 * input: image, output(dest), ONE filter, filter dimensions, input image dimensions, bias
 * Filter and kernel is used interchangeably here as input is only 1 channel and there is only one
 * conv layer in the network. Design mistake by me but it works out thanks to the architecture.
 */
void conv(double** image, double** output, double **filter, int filter_h, int filter_w, int in_h, int in_w, double bias, ActivationFn activation_fn) {

    for (int i = 0; i < in_h - filter_h + 1; i++) { //Again, with stride = 1 and 0 padding this is valid row iteration
        for (int j = 0; j < in_w - filter_w + 1; j++) {
            double sum = 0.0;
            for (int k = 0; k < filter_h; k++) {
                for (int z = 0; z < filter_w; z++) {
                    sum += image[i + k][j + z] * filter[k][z];
                }
            }
            sum += bias;
            output[i][j] = activation_fn(sum);
        }
    }
}


/*
 * inputs: ConvLayer pointer, 2d image array (one image) (double**), height and width of image
 * outputs: feature maps
 */
void conv_forward(ConvLayer *layer, double ***feature_out, double** image, int in_h, int in_w, ActivationFn activation_fn) {
    //double ***output = malloc(sizeof(double**) * layer->numFilters);
    //assert(output);

    // feature map size = inputSize - kernelSize + 1 (formula changes if stride != 1)
    // kernelSize = 3x3, 28-3+1 = 26)
    int out_h = (in_h - layer->filterSize) + 1;   //  stride = 1, padding = 0
    int out_w = (in_w - layer->filterSize) + 1;

    // allocates memory for feature maps, and calls conv with filter[i]
    for (int i = 0; i < layer->numFilters; i++) {
        //output[i] = malloc(sizeof(double*) * out_h);
        //assert(output[i]);

        conv(image, feature_out[i], layer->filters[i], layer->filterSize, layer->filterSize, in_h, in_w, layer->biases[i], activation_fn);
    }

}


void update_conv(ConvLayer *layer, double learning_rate, double beta) {

    for (int i = 0; i < layer->numFilters; i++) {

        for (int j = 0; j < layer->filterSize; j++) {

            for (int k = 0; k < layer->filterSize; k++) {

                layer->vfilters[i][j][k] = beta * layer->vfilters[i][j][k] + layer->dfilters[i][j][k];

                layer->filters[i][j][k] -= learning_rate * layer->vfilters[i][j][k];

            }
        }

        layer->vbiases[i] = beta * layer->vbiases[i] + layer->dbiases[i];

        layer->biases[i] -= learning_rate * layer->vbiases[i];
    }
}

void zero_conv_gradients(ConvLayer *layer) {

    memset(layer->dbiases, 0, sizeof(double) * layer->numFilters);

    for (int i = 0; i < layer->numFilters; i++) {

        for (int j = 0; j < layer->filterSize; j++) {
            memset(layer->dfilters[i][j], 0, sizeof(double) * layer->filterSize);

        }

    }
}


void free_convlayer(ConvLayer *layer) {
    if (!layer) return;

    for (int i = 0; i < layer->numFilters; i++) {

        for (int j = 0; j < layer->filterSize; j++) {
            free(layer->filters[i][j]);
            free(layer->dfilters[i][j]);
            free(layer->vfilters[i][j]);
        }

        free(layer->filters[i]);
        free(layer->dfilters[i]);
        free(layer->vfilters[i]);
    }

    free(layer->filters);
    free(layer->dfilters);
    free(layer->vfilters);

    free(layer->biases);
    free(layer->dbiases);
    free(layer->vbiases);

    free(layer);
}

