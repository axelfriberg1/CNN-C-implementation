//
// Created by Axel Friberg on 2025-12-10.
//


#include "../headers/utils.h"

#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>


void flatten(double ***feature_maps, double *dest, int n, int h, int w) {

    // x + HEIGHT* (y + WIDTH* z): https://stackoverflow.com/questions/7367770/how-to-flatten-or-index-3d-array-in-1d-array
    int idx = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < h; j++) {
            for (int k = 0; k < w; k++) {
                dest[idx++] = feature_maps[i][j][k];
            }
        }
    }
}

void unflatten(double *src, double ***dest, size_t src_size, size_t dest_n, size_t dest_h, size_t dest_w) {
    int idx = 0;
    for (size_t i1 = 0; i1 < dest_n; i1++) {
        for (size_t j = 0; j < dest_h; j++) {
            for (size_t k1 = 0; k1 < dest_w; k1++) {
                dest[i1][j][k1] = src[idx++];
            }
        }
    }
}


double maxValue(double *arr, size_t size) {
    double maxValue = arr[0];

    for (size_t i = 1; i < size; ++i) {
        if ( arr[i] > maxValue ) {
            maxValue = arr[i];
        }
    }
    return maxValue;
}

void zero_3d(double ***src, size_t n, size_t h, size_t w) {
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < h; ++j)
            for (size_t k = 0; k < w; ++k)
                src[i][j][k] = 0.0;
}

double ***allocate_3D(size_t n, size_t h, size_t w, bool zero_init) {
    double ***top = malloc(sizeof(double**) * n);


    for (size_t i = 0; i < n; i++) {
        top[i] = malloc(sizeof(double*) * h);

        for (size_t j = 0; j < h; j++) {
            if (zero_init) {
                top[i][j] = calloc(w, sizeof(double));
            } else {
                top[i][j] = malloc(sizeof(double) * w);
            }
        }
    }

    return top;
}

double **allocate_2D(size_t n, size_t size, bool zero_init) {
    double **top = malloc(sizeof(double*) * n);


    for (size_t i = 0; i < n; i++) {
        if (zero_init) {
            top[i] = calloc(size, sizeof(double));
        } else {
            top[i] = malloc(sizeof(double) * size);
        }
    }

    return top;
}


void free_3d_array(double ***src, int n, int h) {

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < h; j++) {
            free(src[i][j]);
        }
        free(src[i]);
    }
    free(src);
}
