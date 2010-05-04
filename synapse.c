#include "synapse.h"

Synapse *synapse_create(Neuron *input, Neuron *output) {
    Synapse *synapse = (Synapse *)malloc(sizeof(Synapse));
    synapse->input = input;
    synapse->output = output;
    synapse->weight = ((float)rand() / ((float)RAND_MAX / 2.0)) - 1; // (-1.0)-1.0
    return synapse;
}