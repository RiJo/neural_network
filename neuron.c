#include "neuron.h"

void neuron_init(Neuron *neuron) {
    neuron->input = 0.0;
    neuron->output = 0.0;
    neuron->bias = BIAS_INPUT;
    neuron->inputs = NULL;
    neuron->outputs = NULL;
    neuron->count.inputs = 0;
    neuron->count.outputs = 0;
}

/* calculates the (input) value of the neuron */
float neuron_value(Neuron *neuron) {
    if (neuron->count.inputs == 0) {
        // neuron in layer 0 (no inputs)
        return neuron->input;
    }

    Synapse *synapse;
    Neuron *input;
    float value = neuron->bias;
    for (unsigned int i = 0; i < neuron->count.inputs; i++) {
        synapse = neuron->inputs[i];
        input = synapse->input;
        value += (input->output * synapse->weight);
    }
    return value;
}

/* produce the output value of the neuron */
void neuron_fire(Neuron *neuron) {
    neuron->input = neuron_value(neuron);
    neuron->output = neuron_sigmoid(neuron);
}

/* the transfer function */
float neuron_sigmoid(Neuron *neuron) {
    return tanh(neuron->input);
}