//
// Created by Axel Friberg on 2025-12-10.
//

#ifndef CNN_ACTIVATION_H
#define CNN_ACTIVATION_H
#include <stddef.h>


typedef double (*ActivationFn)(double val);

double relu(double val);
void softmax(double *input, double* probs, size_t in_size);
double none(double val);


#endif //CNN_ACTIVATION_H