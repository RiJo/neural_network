/*
 * A neuron is the most important block in the neural network. It handles inputs
 * and with a fire function the output of a neuron is calculated. The neurons
 * in the neural network are connected to each other by synapses. If the
 * synapses represents edges in a graph data structure, then the neurons
 * represents the nodes.
 */

#ifndef _NEURON_H_
#define _NEURON_H_

#include "synapse.h"
#include <math.h>

#ifdef _DEBUG_
#include <assert.h>
#define ASSERT assert
#define DEBUG printf("[debug] ");printf
#else
#define ASSERT(arg1,...)
#define DEBUG(arg1,...)
#endif

#define BIAS_INPUT 1.0

struct Neuron {
    float bias;
    float input; /* dendrites */
    float output; /* axon */

    Synapse **inputs;
    Synapse **outputs;

    unsigned int input_count;
    unsigned int output_count;

    float (*sigmoid_function)(float);
    float (*dsigmoid_function)(float);
};

void neuron_init(Neuron *);
void neuron_destroy(Neuron *);
void neuron_fire(Neuron *);
float neuron_value(Neuron *);
float neuron_sigmoid(Neuron *);
float neuron_dsigmoid(Neuron *);

void neuron_set_sigmoid(Neuron *, float (*)(float));
void neuron_set_dsigmoid(Neuron *, float (*)(float));

#endif
