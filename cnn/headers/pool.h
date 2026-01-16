//
// Created by Axel Friberg on 2025-12-08.
//

#ifndef CNN_POOL_H
#define CNN_POOL_H
#include <stdlib.h>


typedef struct {
    int in_h;
    int in_w;
    int num_fmaps;

    int pool_size;

} PoolingLayer;

PoolingLayer *init_pooling(int in_h, int in_w, int pool_size, int num_fmaps);
void pool(PoolingLayer *layer, double ***pool_maps, double ***feature_maps);
void max_pool(double **feature_map, double **output, int in_h, int in_w);
void free_pool(PoolingLayer *layer);
#endif //CNN_POOL_H