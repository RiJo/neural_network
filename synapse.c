#include "synapse.h"

Synapse *synapse_create(Neuron *input, Neuron *output, float diff) {
    Synapse *synapse = (Synapse *)malloc(sizeof(Synapse));
    if (synapse == NULL) {
        return NULL;
    }
    synapse->input = input;
    synapse->output = output;
    synapse->weight = ((float)rand() / ((float)RAND_MAX / diff)) - (diff / 2);
    synapse->change = 0.0;
    return synapse;
}
