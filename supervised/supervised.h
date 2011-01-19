/*
 * This module ...
 */

#ifndef _SUPERVISED_H_
#define _SUPERVISED_H_

#include "neural_network.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#ifdef _DEBUG_
#include <assert.h>
#define ASSERT assert
#define DEBUG printf("[debug] ");printf
#else
#define ASSERT(arg1,...)
#define DEBUG(arg1,...)
#endif

// train data
typedef struct {
    float **input;
    float **output;
    unsigned int data_count;
    unsigned int input_count;
    unsigned int output_count;
} TD;

TD *td_create(unsigned int, unsigned int);
void td_destroy(TD *);
void td_add(TD *, float *, float *);

// supervised
float nn_error_factor(NN *, TD *);
float nn_train(NN *, TD *);

#endif
