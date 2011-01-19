#include "synapse.h"

Synapse *synapse_create(Neuron *input, Neuron *output, float diff) {
    ASSERT(input);
    ASSERT(output);

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
    ASSERT(synapse);

    free(synapse);
}

void synapse_change(Synapse *synapse, float value) {
    ASSERT(synapse);

    synapse->weight += (value * synapse->learn_rate) + (synapse->change * synapse->momentum);
    //if (synapse->weight  < 0.0) // TODO: is it correct to place it here?
    //    synapse->weight = 0.0; // TODO: synapse should be removed from network...
    synapse->change = value;
}
