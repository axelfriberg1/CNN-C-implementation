//
// Created by Axel Friberg on 2025-12-19.
//

#include "../headers/backprop.h"

#include <string.h>

#include "../headers/dense.h"
#include "../headers/pool.h"
#include "../headers/convlayer.h"

#include <tgmath.h>


double softmax_cross_entropy_back(double *preds, int true_label, double *dlogits, int num_classes) {

    for (int i = 0; i < num_classes; i++) {
        dlogits[i] = preds[i];
    }

    // Due to simplifications (seen here: https://www.youtube.com/watch?v=6ArSys5qHAU)
    // For softmax + cross-entropy, the gradient of the loss
    // with respect to the logits simplifies to pi - yi
    // (yi = 1 for the true class, 0 for all others)
    // https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
    dlogits[true_label] -= 1.0;

    // Cross-entropy loss
    return -log(preds[true_label]);


}

double cross_entropy(double *preds, int true_label, int num_classes) {

    // Cross-entropy loss
    return -log(preds[true_label]);


}

void dense_back_relu(
    DenseLayer *layer,
    double *layer_out,
    const double *input,
    const double *dout, // upstream error / error signal
    double *dinput) {


    int in_size  = layer->in_size;
    int out_size = layer->out_size;
    double dz[out_size];

    // Relu gate for the neurons
    for (int i = 0; i < out_size; i++) {
        if (layer_out[i] <= 0) {
            dz[i] = 0.0;
        } else {
            dz[i] = dout[i];
        }
    }

    // Backprop
    for (int i = 0; i < out_size; i++) {

        // Bias gradient
        layer->dbiases[i] += dz[i];

        for (int j = 0; j < in_size; j++) {

            // Weight gradient
            layer->dweights[i][j] += dz[i] * input[j];


            dinput[j] += layer->weights[i][j] * dz[i];
        }
    }

}

void dense_back_no_activation(
    DenseLayer *layer,
    const double *input,     // previous activation (size: in_size)
    const double *dlogits,   // dL / dz (size: out_size)
    double *dinput           // dL / dinput (size: in_size)
) {

    // due to simplification made by softmax + cross-entropy
    // this reduces to (p^i - y^i) * a^j or here: dlogits[i] * weight[o][i]
    // for bias: dbias[i] = dlogits[i] since dL / db = 1

    int in_size  = layer->in_size;
    int out_size = layer->out_size;


    // Zero parameter gradients
    /*for (int i = 0; i < out_size; i++) {
        layer->dbiases[i] = 0.0;
        for (int j = 0; j < in_size; j++) {
            layer->dweights[i][j] = 0.0;
        }
    }*/

    // Backprop
    for (int i = 0; i < out_size; i++) {

        // Bias gradient
        layer->dbiases[i] += dlogits[i];

        for (int j = 0; j < in_size; j++) {

            // Weight gradient
            layer->dweights[i][j] += dlogits[i] * input[j];

            // Input gradient to be passed to prev layer, simple since no activation function was used here.
            dinput[j] += layer->weights[i][j] * dlogits[i];
        }
    }
}


void max_pool_back(double **feature_map, double **pooled_images, double **dout, double **dinput) {

    // reverse max_pool
    // loop over each pooled pixel and check for match in the corresponding input feature map.
    for (int i = 0; i < 13; i++) {
        for (int j = 0; j < 13; j++) {

            // I did not cache index of argmax so we recompute it in the 2×2 window
            int max_k = 0, max_z = 0;
            double max_val = feature_map[i * 2][j * 2];

            for (int k = 0; k < 2; k++) {
                for (int z = 0; z < 2; z++) {
                    double v = feature_map[i * 2 + k][j * 2 + z];
                    if (v > max_val) {
                        max_val = v;
                        max_k = k;
                        max_z = z;
                    }
                }
            }

            // Route gradient to exactly one position
            dinput[i * 2 + max_k][j * 2 + max_z] += dout[i][j];
        }
    }

}


void pool_back(PoolingLayer *layer, double ***feature_maps, double*** pooled_images, double ***dout, double ***dinput) {
    // for max pooling dL / dInput = 0, for all non-max values in each region, ie no effect on the loss.

    int out_h = layer->in_h/2;
    int out_w = layer->in_w/2;



    // iterate and pool
    for (int i = 0; i < layer->num_fmaps; i++) {
        max_pool_back(feature_maps[i], pooled_images[i], dout[i], dinput[i]);
    }

}




void conv_back(
    ConvLayer *layer,
    double **input_image,        // [28][28]
    double ***feature_maps,      // [6][26][26] (after ReLU)
    double ***dout,              // [6][26][26] (from pooling back)
    double ***dconv              // [6][28][28] (output buffer)
)
{

    int F = layer->filterSize;          // 3
    int out_h = 28 - F + 1;              // 26
    int out_w = 28 - F + 1;              // 26

    // looping over each filter
    for (int f = 0; f < layer->numFilters; f++) {

        // loop over feature map locations
        for (int i = 0; i < out_h; i++) {
            for (int j = 0; j < out_w; j++) {

                // ReLU backward gate
                if (feature_maps[f][i][j] <= 0.0) {
                    continue;
                }

                double grad = dout[f][i][j];

                // bias gradient (one bias per filter)
                layer->dbiases[f] += grad;

                // loop over filter window
                for (int u = 0; u < F; u++) {
                    for (int v = 0; v < F; v++) {

                        // gradient w.r.t filter weight
                        layer->dfilters[f][u][v] +=
                            input_image[i + u][j + v] * grad;

                        // gradient w.r.t input image
                        // technically I only need this if I add more conv_layers since this is
                        // the error signal that propagates back.
                        dconv[f][i + u][j + v] +=
                            layer->filters[f][u][v] * grad;
                    }
                }
            }
        }
    }
}