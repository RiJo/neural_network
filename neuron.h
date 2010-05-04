/*
 * A neuron is the most important block in the neural network. It handles inputs
 * and with a fire function the output of a neuron is calculated. The neurons
 * in the neural network are connected to each other by synapses. If the
 * synapses represents edges in a graph data structure, then the neurons
 * represents the nodes.
 */

#ifndef _NEURON_H_
#define _NEURON_H_

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "synapse.h"

#define BIAS_INPUT 1.0

/*
Artificial Neuron
The artificial neuron receives one or more inputs (representing the one or more dendrites) and sums them to produce an output (synapse). Usually the sums of each node are weighted, and the sum is passed through a non-linear function known as an activation function or transfer function. The transfer functions usually have a sigmoid shape, but they may also take the form of other non-linear functions, piecewise linear functions, or step functions. They are also often monotonically increasing, continuous, differentiable and bounded.
*/

struct Neuron {
    float bias;
    float input;
    float output;
    float last_change;

    Synapse **inputs;
    Synapse **outputs;

    unsigned int input_count;
    unsigned int output_count;
};

void neuron_init(Neuron *);
void neuron_destroy(Neuron *);
void neuron_fire(Neuron *);
float neuron_value(Neuron *);
float neuron_sigmoid(Neuron *);
float neuron_dsigmoid(Neuron *);


#endif
