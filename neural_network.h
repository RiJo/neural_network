#ifndef _NEURAL_NETWORK_H_
#define _NEURAL_NETWORK_H_

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <stdio.h>
#include "neuron.h"
#include "train_data.h"

/*
    Notes:
        - rename output/input of neuron (cunfusion with the same values of synapse)
        - fix variable names!! especially in for loops!!
        - fix error_factor() and train() return values
        - printout invalid mallocs()
*/

typedef struct {
    Neuron **layers;
    Synapse **synapses;
    unsigned int layer_count;
    unsigned int *neuron_count;
    unsigned int synapse_count;
} NN;

NN *nn_create(unsigned int, unsigned int *);
void nn_destroy(NN *);
void nn_add_synapse(NN *, Neuron *, Neuron *);
void nn_generate_synapses(NN *);
void nn_set_input(NN *, unsigned int, float);
float nn_read_output(NN *, unsigned int);
void nn_calculate(NN *);
float nn_error_factor(NN *, TD *);
float nn_train(NN *, TD *, float, float);

#endif