/*
 * A synapse represents a link between two neurons. It is made bidirectional
 * to handle backpropagation in a simple way. If the neurons represents nodes
 * in a graph data structure, then the synapses represents the edges.
 */

#ifndef _SYNAPSE_H_
#define _SYNAPSE_H_

#include <stdint.h>
#include <stdlib.h>
#include <math.h>

typedef struct Neuron Neuron; // forward declaration

typedef struct {
    Neuron *input;
    Neuron *output;
    float weight; /* 0 if no connection */
    float last_change;
} Synapse;

Synapse *synapse_create(Neuron *, Neuron *);

#endif
