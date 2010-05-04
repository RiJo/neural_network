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
*/


typedef struct {
    Neuron **layers;
    Synapse **synapses;
    unsigned int layer_count;
    unsigned int *neuron_count;
    unsigned int synapse_count;
} NN;

NN *create_neural_network(unsigned int, unsigned int *);
void destroy_neural_network(NN *);

void add_synapse(NN *, Neuron *, Neuron *);
void generate_synapses(NN *);

void set_input(NN *, unsigned int, float);
float read_output(NN *, unsigned int);
void calculate(NN *);
float error_factor(NN *, TD *);
float train(NN *, TD *, float, float);

#endif