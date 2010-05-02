#ifndef _SYNAPSE_H_
#define _SYNAPSE_H_

#include <stdint.h>
#include <math.h>

typedef struct Neuron Neuron; // forward declaration

typedef struct {
    Neuron *input;
    Neuron *output;
    float weight; /* 0 if no connection */
} Synapse;

#endif