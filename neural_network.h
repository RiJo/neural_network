/*
 * The neural network consists of a dynamic number of layers (usually three) in
 * which there exists a dynamic number of neurons. The layers (group of neurons)
 * are connected by synapses. Data is put into the first layer's neurons inputs
 * and are then calculated to the last layer's neurons outputs.
 * This network uses backpropagation for training. This means that training
 * data (list of inputs with corresponding outputs) can be passed to the neural
 * network which is re-calibrated accordingly.
 */
 
 /*
    Fixes:
        - rename output/input of neuron (cunfusion with the same values of synapse)
*/

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