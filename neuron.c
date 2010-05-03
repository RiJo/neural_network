#include "neuron.h"

void neuron_init(Neuron *neuron) {
    neuron->input = 0.0;
    neuron->output = 0.0;
    neuron->last_change = 0.0;
    neuron->bias = BIAS_INPUT;
    neuron->inputs = NULL;
    neuron->outputs = NULL;
    neuron->input_count = 0;
    neuron->output_count = 0;
}

void neuron_destroy(Neuron *neuron) {
    free(neuron->inputs);
    free(neuron->outputs);
    //free(neuron);
}

/* calculates the (input) value of the neuron */
float neuron_value(Neuron *neuron) {
    if (neuron->input_count == 0) {
        // neuron in layer 0 (no inputs)
        return neuron->input;
    }

    Synapse *synapse;
    Neuron *input;
    float value = neuron->bias;
    for (unsigned int i = 0; i < neuron->input_count; i++) {
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

/* derivative of the sigmoid function */
float neuron_dsigmoid(Neuron *neuron) {
    return 1.0 - pow(neuron->output, 2);
}