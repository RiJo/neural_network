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
    synapse->momentum = 0.5;
    synapse->learn_rate = 0.5;

    return synapse;
}

void synapse_destroy(Synapse *synapse) {
#ifdef DEBUG
    assert(synapse);
#endif

    free(synapse);
}

void synapse_change(Synapse *synapse, float value) {
#ifdef DEBUG
    assert(synapse);
#endif

    synapse->weight += (value * synapse->learn_rate) + (synapse->change * synapse->momentum);
    synapse->change = value;
}
