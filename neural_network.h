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
 *    [NN-1.0<2:5:2>]
 *    layer1:n1:layer2:n2:weight:change
 *    layer1:n1:layer2:n2:weight:change
 *    layer1:n1:layer2:n2:weight:change
 */
 
 /*
    Fixes:
        - rename output/input of neuron (cunfusion with the same values of synapse)
        - add comment and statistics to dump file (easier to see what kind of data)
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

#define NN_FILE_DUMP_VERSION "1.1"

typedef struct {
    Neuron **layers;
    Synapse **synapses;
    unsigned int layer_count;
    unsigned int *neuron_count;
    unsigned int synapse_count;
    char *comment;
} NN;

NN *nn_create(unsigned int, unsigned int *);
NN *nn_load_from_file(FILE *);
void nn_dump_to_file(NN *, FILE *, char *);
void nn_destroy(NN *);
void nn_add_synapse(NN *, unsigned int, unsigned int, unsigned int, unsigned int);
void nn_generate_synapses(NN *);
void nn_set_input(NN *, unsigned int, float);
float nn_read_output(NN *, unsigned int);
void nn_calculate(NN *);
float nn_error_factor(NN *, TD *);
float nn_train(NN *, TD *, float, float);

#endif
