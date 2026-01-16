//
// Created by Axel Friberg on 2025-12-31.
//

#include "../headers/cnn.h"

#include <string.h>

#include "../headers/utils.h"


Cnn *init_cnn(
    ConvLayer *conv1,
    PoolingLayer *pool1,
    DenseLayer *dense1,
    DenseLayer *dense2,
    DenseLayer *dense3,
    size_t img_h,
    size_t img_w) {

    Cnn *cnn = malloc(sizeof(Cnn));
    assert(cnn);

    cnn->conv1 = conv1;
    cnn->pool1 = pool1;
    cnn->dense1 = dense1;
    cnn->dense2 = dense2;
    cnn->dense3 = dense3;

    cnn->img_h = img_h;
    cnn->img_w = img_w;


    return cnn;

}

CnnMemory *init_memory(Cnn *cnn) {

    CnnMemory *memory = malloc(sizeof(CnnMemory));

    // Forward buffers ======================================================
    memory->conv1_buffer = allocate_3D(
        cnn->conv1->numFilters,
        cnn->img_h - cnn->conv1->filterSize + 1,
        cnn->img_w - cnn->conv1->filterSize + 1,
        false);
    memory->pool1_buffer = allocate_3D(
        cnn->pool1->num_fmaps,
        cnn->pool1->in_h/2,
        cnn->pool1->in_w/2,
        false);
    memory->flat_buffer = malloc(sizeof(double) * cnn->pool1->in_h/2 * cnn->pool1->in_w/2);
    memory->dense1_buffer = malloc(sizeof(double) * cnn->dense1->out_size);
    memory->dense2_buffer = malloc(sizeof(double) * cnn->dense2->out_size);
    memory->dense3_buffer = malloc(sizeof(double) * cnn->dense3->out_size);
    memory->probs_buffer = malloc(sizeof(double) * cnn->dense3->out_size);

    // // Backward buffers =====================================================================
    // naming schema here is grad_xxx
    // where xxx represents w.r.t xxx_output
    memory->grad_logits = malloc(sizeof(double) * cnn->dense3->out_size);
    memory->grad_dense2= malloc(sizeof(double) * cnn->dense2->out_size);
    memory->grad_dense1 = malloc(sizeof(double) * cnn->dense1->out_size);
    memory->grad_flat = malloc(sizeof(double) * cnn->dense1->in_size);
    memory->grad_pool = allocate_3D(
        cnn->pool1->num_fmaps,
        cnn->pool1->in_h/2,
        cnn->pool1->in_w/2,
        true);
    memory->grad_conv = allocate_3D(
        cnn->pool1->num_fmaps,
        cnn->img_h - cnn->conv1->filterSize + 1,
        cnn->img_w - cnn->conv1->filterSize + 1,
        true);
    memory->grad_input = allocate_3D(
        cnn->conv1->numFilters,
        cnn->img_h,
        cnn->img_w,
        true);


    return memory;

}

void cnn_zero_forward_buffers(Cnn *cnn, CnnMemory *memory) {
    zero_3d(memory->conv1_buffer, cnn->conv1->numFilters, cnn->img_h - cnn->conv1->filterSize + 1, cnn->img_w - cnn->conv1->filterSize + 1); // 6 * 26 * 26
    zero_3d(memory->pool1_buffer, cnn->pool1->num_fmaps, cnn->pool1->in_h/2, cnn->pool1->in_w/2); // 6 * 13 * 13
    memset(memory->flat_buffer, 0, sizeof(double) * cnn->pool1->num_fmaps * cnn->pool1->in_h/2 * cnn->pool1->in_w/2);
    memset(memory->dense1_buffer, 0, sizeof(double) * cnn->dense1->out_size);
    memset(memory->dense2_buffer, 0, sizeof(double) * cnn->dense2->out_size);
    memset(memory->dense3_buffer, 0, sizeof(double) * cnn->dense3->out_size);
    memset(memory->probs_buffer, 0, sizeof(double) * cnn->dense3->out_size);
}
void cnn_zero_backward_buffers(Cnn *cnn, CnnMemory *memory) {

    memset(memory->grad_logits, 0, sizeof(double) * cnn->dense3->out_size);
    memset(memory->grad_dense2, 0, sizeof(double) * cnn->dense2->out_size);
    memset(memory->grad_dense1, 0, sizeof(double) * cnn->dense1->out_size);
    memset(memory->grad_flat, 0, sizeof(double) * cnn->pool1->num_fmaps * (cnn->pool1->in_h / 2) * (cnn->pool1->in_w / 2));

    zero_3d(memory->grad_pool,
            cnn->pool1->num_fmaps,
            cnn->pool1->in_h / 2,
            cnn->pool1->in_w / 2);
    zero_3d(memory->grad_conv,
            cnn->conv1->numFilters,
            cnn->img_h - cnn->conv1->filterSize + 1,
            cnn->img_w - cnn->conv1->filterSize + 1);
    zero_3d(memory->grad_input,
            cnn->conv1->numFilters,
            cnn->img_h,
            cnn->img_w);

}


void free_memory(CnnMemory *memory, Cnn *cnn) {
    if (!memory) return;

    /* Forward buffers */
    free_3d_array(
        memory->conv1_buffer,
        cnn->conv1->numFilters,
        cnn->img_h - cnn->conv1->filterSize + 1
    );

    free_3d_array(
        memory->pool1_buffer,
        cnn->pool1->num_fmaps,
        cnn->pool1->in_h / 2
    );

    free(memory->flat_buffer);
    free(memory->dense1_buffer);
    free(memory->dense2_buffer);
    free(memory->dense3_buffer);
    free(memory->probs_buffer);

    /* Backward buffers */
    free(memory->grad_logits);
    free(memory->grad_dense2);
    free(memory->grad_dense1);
    free(memory->grad_flat);

    free_3d_array(
        memory->grad_pool,
        cnn->pool1->num_fmaps,
        cnn->pool1->in_h / 2
    );

    free_3d_array(
        memory->grad_conv,
        cnn->conv1->numFilters,
        cnn->img_h - cnn->conv1->filterSize + 1
    );

    free_3d_array(
        memory->grad_input,
        cnn->conv1->numFilters,
        cnn->img_h
    );

    free(cnn);
    free(memory);
}
