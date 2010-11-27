#include "synapse.h"

Synapse *synapse_create(Neuron *input, Neuron *output, float diff) {
#ifdef DEBUG
    assert(input);
    assert(output);
#endif

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

void synapse_destroy(Synapse *synapse) {
#ifdef DEBUG
    assert(synapse);
#endif

    free(synapse);
}
