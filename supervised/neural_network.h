/*
 * The neural network consists of a dynamic number of layers (usually three) in
 * which there exists a dynamic number of neurons. The layers (group of neurons)
 * are connected by synapses. Data is put into the first layer's neurons inputs
 * and are then calculated to the last layer's neurons outputs.
 * This network uses backpropagation for training. This means that training
 * data (list of inputs with corresponding outputs) can be passed to the neural
 * network which is re-calibrated accordingly.
 *
 * Structure of dump file:
 *    [NN-1.1<2:5:2>]
 *    space for comment
 *    layer1:n1:layer2:n2:weight:change
 *    layer1:n1:layer2:n2:weight:change
 *    layer1:n1:layer2:n2:weight:change
 */

#ifndef _NEURAL_NETWORK_H_
#define _NEURAL_NETWORK_H_

#include "neuron.h"
#include <stdio.h>
#include <string.h>
#ifdef DEBUG
#include <assert.h>
#endif

#define NN_FILE_DUMP_VERSION "1.1"

typedef struct {
    Neuron **layers;
    Synapse **synapses;
    unsigned int layer_count;
    unsigned int *neuron_count;
    unsigned int synapse_count;
} NN;

NN *nn_create(unsigned int, unsigned int *);
NN *nn_load_from_file(FILE *);
void nn_dump_to_file(NN *, FILE *);
void nn_destroy(NN *);
size_t nn_size(NN *);
//void nn_add_neuron(); /* not implemented */
void nn_add_synapse(NN *, unsigned int, unsigned int, unsigned int, unsigned int);
//void nn_remove_neuron(); /* not implemented */
//void nn_remove_synapse(); /* not implemented */
void nn_generate_synapses(NN *); /* generates a fully connected network */
void nn_set_input(NN *, unsigned int, float);
float nn_read_output(NN *, unsigned int);
void nn_calculate(NN *);
int nn_connected(NN *); /* TODO: also create a nn_fully_connected(NN*) function */

#endif
