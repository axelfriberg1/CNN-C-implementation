//
// Created by Axel Friberg on 2025-12-08.
//

# include "../headers/pool.h"

#include <assert.h>
#include <stdlib.h>


PoolingLayer *init_pooling(int in_h, int in_w, int pool_size, int num_fmaps) {

    PoolingLayer *layer = malloc(sizeof(PoolingLayer));
    assert(layer);

    layer->in_h = in_h;
    layer->in_w = in_w;
    layer->pool_size = pool_size;
    layer->num_fmaps = num_fmaps;

    return layer;
}

void max_pool(double **feature_map, double **output, int in_h, int in_w) {

    // NOTE, assume we use pool size of 2x2 and we don't overlap like in convolution.
    for (int i = 0; i < in_h; i += 2) {
        for (int j = 0; j < in_w; j += 2) {
            double max = feature_map[i][j];

            for (int k = 0; k < 2; k++) {
                for (int z = 0; z < 2; z++) {
                    if (feature_map[i+k][j+z] > max) {
                        max = feature_map[i+k][j+z];
                    }
                }
            }

            output[i/2][j/2] = max;
        }
    }
}

void pool(PoolingLayer *layer, double*** pool_maps, double ***feature_maps) {

    //double ***output = malloc(sizeof(double**) * layer->num_fmaps);
    //assert(output);

    int out_h = layer->in_h/2;
    int out_w = layer->in_w/2;



    // iterate and pool.
    for (int i = 0; i < layer->num_fmaps; i++) {
        //output[i] = malloc(sizeof(double*) * out_h);
        for (int j = 0; j < out_h; j++) {
            //output[i][j] = malloc(sizeof(double) * out_w);
        }
        max_pool(feature_maps[i], pool_maps[i], layer->in_h, layer->in_w);
    }
}



void free_pool(PoolingLayer *layer) {
    if (!layer) return;
    free(layer);
}
