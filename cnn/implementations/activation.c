//
// Created by Axel Friberg on 2025-12-10.
//


#include <assert.h>
#include <stddef.h>
#include <stdlib.h>
#include <tgmath.h>
#include "../headers/utils.h"

double relu(double val) {
    return (val > 0) ? val : 0;
}

void softmax(double *input, double *probs, size_t in_size) {
    double max = maxValue(input, in_size);
    double sum = 0.0;
    //double *output = malloc(sizeof(double) * in_size);

    // formula with numerical stability found here: https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
    // should shift inputs to a range close to zero and shifts them all to negative ("negatives with large exponents saturate to zero")
    // and have a better chance of avoiding inf.
    // I don't fully understand this yet.
    for (size_t i = 0; i < in_size; i++) {
        double expo = exp(input[i] - max);
        sum += expo;
        probs[i] = expo;
    }

    assert(sum != 0);

    for (size_t i = 0; i < in_size; i++) {
        probs[i] /= sum;
    }
}

double none(double val) {
    return val;
}