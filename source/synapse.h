/*
 * A synapse represents a link between two neurons. It is made bidirectional
 * to handle backpropagation in a simple way. If the neurons represents nodes
 * in a graph data structure, then the synapses represents the edges.
 */

#ifndef _SYNAPSE_H_
#define _SYNAPSE_H_

#include <stdlib.h>

#ifdef _DEBUG_
#include <assert.h>
#define ASSERT assert
#define DEBUG printf("[debug] ");printf
#else
#define ASSERT(arg1,...)
#define DEBUG(arg1,...)
#endif

typedef struct Neuron Neuron; // forward declaration

typedef struct {
    Neuron *input;
    Neuron *output;
    float weight; /* 0 if no connection */
    float change; /* last change in weight */
    float momentum; /* how much the last change will affect the new one */
    float learn_rate; /* the intensity of the change value */
} Synapse;

Synapse *synapse_create(Neuron *, Neuron *, float);
void synapse_destroy(Synapse *);
void synapse_change(Synapse *, float);

#endif
