#ifndef _TRAIN_DATA_H_
#define _TRAIN_DATA_H_

#include <stdlib.h>
#include <string.h>

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