#include "neural_network.h"

/* Initialize the neural network with all neurons */
NN *nn_create(unsigned int layers, unsigned int *neurons) {
    srand (time(NULL));

    // allocate memory
    NN *network = (NN *)malloc(sizeof(NN));
    network->layers = (Neuron **)malloc(sizeof(Neuron *) * layers);
    for (unsigned int layer = 0; layer < layers; layer++) {
        network->layers[layer] = (Neuron *)malloc(sizeof(Neuron) * neurons[layer]);
        for (unsigned int neuron = 0; neuron < neurons[layer]; neuron++) {
            neuron_init(&network->layers[layer][neuron]);
        }
    }

    network->layer_count = layers;
    network->synapses = NULL;
    network->synapse_count = 0;
    network->neuron_count = (unsigned int *)malloc(sizeof(unsigned int) * layers);
    memcpy(network->neuron_count, neurons, sizeof(unsigned int) * layers);

    return network;
}

/* Free all memory allocated for the neural network */
void nn_destroy(NN *network) {
    for (unsigned int layer = 0; layer < network->layer_count; layer++) {
        for (unsigned int neuron = 0; neuron < network->neuron_count[layer]; neuron++) {
            neuron_destroy(&(network->layers[layer][neuron]));
        }
        free(network->layers[layer]); // free neuron block
    }
    free(network->layers);
    free(network->neuron_count);
    for (unsigned int synapse = 0; synapse < network->synapse_count; synapse++) {
        free(network->synapses[synapse]);
    }
    free(network->synapses);
    free(network);
}

/* Generates a synapse between the given neurons */
void nn_add_synapse(NN *network, Neuron *input, Neuron *output) {
    assert(network);
    assert(input);
    assert(output);

    Synapse *synapse = synapse_create(input, output);

    // store synapse reference
    network->synapse_count++;
    network->synapses = (Synapse **)realloc(network->synapses, sizeof(Synapse *) * network->synapse_count);
    network->synapses[network->synapse_count - 1] = synapse;

    // bind input neuron
    input->output_count++;
    input->outputs = (Synapse **)realloc(input->outputs, sizeof(Synapse *) * input->output_count);
    input->outputs[input->output_count - 1] = synapse;

    // bind output neuron
    output->input_count++;
    output->inputs = (Synapse **)realloc(output->inputs, sizeof(Synapse *) * output->input_count);
    output->inputs[output->input_count - 1] = synapse;
}

/* Generate a synapse between all neurons at the border between the layers */
void nn_generate_synapses(NN *network) {
    assert(network);

    for (unsigned int layer = 1; layer < network->layer_count; layer++) {
        for (unsigned int neuron1 = 0; neuron1 < network->neuron_count[layer - 1]; neuron1++) {
            for (unsigned int neuron2 = 0; neuron2 < network->neuron_count[layer]; neuron2++) {
                nn_add_synapse(network, &network->layers[layer - 1][neuron1], &network->layers[layer][neuron2]);
            }
        }
    }
}

/* Sets an input neuron to a given value */
void nn_set_input(NN *network, unsigned int index, float value) {
    assert(network);
    assert(index < network->neuron_count[0]);

    network->layers[0][index].input = value;
}

/* Reads the current value of the given output neuron */
float nn_read_output(NN *network, unsigned int index) {
    assert(network);
    assert(index < network->layer_count);

    return network->layers[network->layer_count - 1][index].output;
}

/* Recalculate the neural network and set the out put neurons dependent on the
    states of the input neurons */
void nn_calculate(NN *network) {
    assert(network);

    for (unsigned int layer = 0; layer < network->layer_count; layer++) {
        for (unsigned int neuron = 0; neuron < network->neuron_count[layer]; neuron++) {
            neuron_fire(&network->layers[layer][neuron]);
        }
    }
}

/* Returns the current error factor of the neural network */
float nn_error_factor(NN *network, TD *train_data) {
    assert(network);
    assert(train_data);

    float error = 0.0;
    for (unsigned int test = 0; test < train_data->data_count; test++) {
        // set inputs
        for (unsigned int input = 0; input < train_data->input_count; input++) {
            network->layers[0][input].input = train_data->input[test][input];
        }
        nn_calculate(network);
        // calculate error in output
        for (unsigned int output = 0; output < train_data->output_count; output++) {
            error += fabs(train_data->output[test][output] - network->layers[network->layer_count - 1][output].output);
        }
    }
    return error / (train_data->output_count * train_data->data_count);
}

/* Backpropagates the network hidden layers recursively */
void backpropagate_hidden(NN *network, float *previous_deltas, float learning_factor, float momentum, unsigned int layer) {
    if (layer == 0) {
        return; // base case
    }

    Neuron *current_neuron;
    float error, change;
    float *deltas = (float *)malloc(sizeof(float) * network->neuron_count[layer]);
    for (unsigned int current = 0; current < network->neuron_count[layer]; current++) {
        current_neuron = &network->layers[layer][current];
        error = 0.0;
        for (unsigned int previous = 0; previous < network->neuron_count[layer + 1]; previous++) {
            error += previous_deltas[previous] * current_neuron->output;
        }
        deltas[current] = error * neuron_dsigmoid(current_neuron);
        for (unsigned int input = 0; input < current_neuron->input_count; input++) {
            change = current_neuron->inputs[input]->input->output * deltas[current];
            current_neuron->inputs[input]->weight += (change * learning_factor) + (current_neuron->last_change * momentum);
            current_neuron->last_change = change;
        }
    }

    // recurse
    backpropagate_hidden(network, deltas, learning_factor, momentum, layer - 1);
    free(deltas);
}

/* Backpropagates the network outputs */
void backpropagate_output(NN *network, TD *train_data, float learning_factor, float momentum) {
    assert(network);
    assert(train_data);

    unsigned int layer = network->layer_count - 1;

    Neuron *current_neuron;
    float error, change;
    float *deltas;
    for (unsigned int test = 0; test < train_data->data_count; test++) {
        // set inputs
        for (unsigned int input = 0; input < train_data->input_count; input++) {
            network->layers[0][input].input = train_data->input[test][input];
        }
        nn_calculate(network);
        // backpropagate
        deltas = (float *)malloc(sizeof(float) * network->neuron_count[layer]);
        for (unsigned int neuron = 0; neuron < network->neuron_count[layer]; neuron++) {
            current_neuron = &network->layers[layer][neuron];
            error = train_data->output[test][neuron] - current_neuron->output;
            deltas[neuron] = error * neuron_dsigmoid(current_neuron);
            for (unsigned int input = 0; input < current_neuron->input_count; input++) {
                change = current_neuron->inputs[input]->input->output * deltas[neuron];
                current_neuron->inputs[input]->weight += (change * learning_factor) + (current_neuron->last_change * momentum);
                current_neuron->last_change = change;
            }
        }
        // recurse
        backpropagate_hidden(network, deltas, learning_factor, momentum, layer - 1);
        free(deltas);
    }
}

/* train the neural network with the defined data */
float nn_train(NN *network, TD *train_data, float learning_factor, float momentum) {
    assert(network);
    assert(train_data);

    float error = nn_error_factor(network, train_data);
    backpropagate_output(network, train_data, learning_factor, momentum);
    return nn_error_factor(network, train_data) - error;
}