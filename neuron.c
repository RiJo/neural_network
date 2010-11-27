#include "neuron.h"

// forward declaration of private functions
float sigmoid(float);
float dsigmoid(float);

void neuron_init(Neuron *neuron) {
#ifdef DEBUG
    assert(neuron);
#endif

    neuron->input = 0.0;
    neuron->output = 0.0;
    neuron->bias = BIAS_INPUT;
    neuron->inputs = NULL;
    neuron->outputs = NULL;
    neuron->input_count = 0;
    neuron->output_count = 0;
    neuron->sigmoid_function = sigmoid;
    neuron->dsigmoid_function = dsigmoid;
}

/* This function differ from other *_destroy(), because it does not free the
   given parameter. Either fix it to handle it, or rename the function. */
void neuron_destroy(Neuron *neuron) {
#ifdef DEBUG
    assert(neuron);
#endif

    free(neuron->inputs);
    neuron->inputs = NULL;
    free(neuron->outputs);
    neuron->outputs = NULL;

    /* this is done from outside, all neurons are allocated by realloc and is
       therefore allocated in one block! */
    //free(neuron);
}

/* calculates the (input) value of the neuron */
float neuron_value(Neuron *neuron) {
#ifdef DEBUG
    assert(neuron);
#endif

    Synapse *synapse;
    float value = neuron->bias;
    for (unsigned int input = 0; input < neuron->input_count; input++) {
        synapse = neuron->inputs[input];
        value += (synapse->input->output * synapse->weight);
    }
    return value;
}

/* produce the output value of the neuron. If it is an input neuron then the
   output value becomes its input value */
void neuron_fire(Neuron *neuron) {
#ifdef DEBUG
    assert(neuron);
#endif

    if (neuron->input_count == 0) {
        neuron->output = neuron->input;
    }
    else {
        neuron->input = neuron_value(neuron);
        neuron->output = neuron_sigmoid(neuron);
    }
}

/* the transfer function */
float neuron_sigmoid(Neuron *neuron) {
#ifdef DEBUG
    assert(neuron);
    assert(neuron->sigmoid_function);
#endif

    return neuron->sigmoid_function(neuron->input);
}

/* derivative of the sigmoid function */
float neuron_dsigmoid(Neuron *neuron) {
#ifdef DEBUG
    assert(neuron);
    assert(neuron->dsigmoid_function);
#endif

    return neuron->dsigmoid_function(neuron->output);
}

/* set custom sigmoid function of neuron */
void neuron_set_sigmoid(Neuron *neuron, float (*custom_sigmoid)(float)) {
#ifdef DEBUG
    assert(neuron);
#endif

    if (custom_sigmoid != NULL)
        neuron->sigmoid_function = custom_sigmoid;
    else
        neuron->sigmoid_function = sigmoid; // restore
}

/* set custom derivative sigmoid function of neuron */
void neuron_set_dsigmoid(Neuron *neuron, float (*custom_dsigmoid)(float)) {
#ifdef DEBUG
    assert(neuron);
#endif

    if (custom_dsigmoid != NULL)
        neuron->dsigmoid_function = custom_dsigmoid;
    else
        neuron->dsigmoid_function = dsigmoid; // restore
}

/* the default sigmoid function */
float sigmoid(float value) {
    return tanh(value);
}

/* the default derivative of sigmoid function */
float dsigmoid(float value) {
    return 1.0 - pow(value, 2);
}
