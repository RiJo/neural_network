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

TD *create_train_data(unsigned int, unsigned int);
void destroy_train_data(TD *);

void add_train_data(TD *, float *, float *);

#endif