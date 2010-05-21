/*
 * This module is created to make a simple interface for training data used to
 * train the neural network. A TD struct is created and then filled with the
 * inputs and corresponding outputs. A pointer to the training data structure
 * is then passed to the neural network's learning function.
 */

#ifndef _TRAIN_DATA_H_
#define _TRAIN_DATA_H_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

typedef struct {
    float **input;
    float **output;
    unsigned int data_count;
    unsigned int input_count;
    unsigned int output_count;
} TD;

TD *train_data_create(unsigned int, unsigned int);
void train_data_destroy(TD *);
void train_data_add(TD *, float *, float *);

#endif
