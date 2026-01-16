//
// Created by Axel Friberg on 2025-12-10.
//

#ifndef CNN_UTILS_H
#define CNN_UTILS_H
#include <stdbool.h>
#include <stddef.h>


void flatten(double ***feature_maps, double *dest, int n, int h, int w);
void unflatten(double *src, double ***dest, size_t src_size, size_t dest_n, size_t dest_h, size_t dest_w);
double maxValue(double *arr, size_t size);
double ***allocate_3D(size_t n, size_t h, size_t w, bool zero_init);
double **allocate_2D(size_t n, size_t size, bool zero_init);
void zero_3d(double ***src, size_t n, size_t h, size_t w);
void free_3d_array(double ***src, int n, int h);

#endif //CNN_UTILS_H