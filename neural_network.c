#include "neural_network.h"

/* Initialize the neural network with all neurons */
NN *create_neural_network(unsigned int layers, unsigned int *neurons) {
    srand (time(NULL));

    // allocate memory
    NN *network = (NN *)malloc(sizeof(NN));
    network->layers = (Neuron **)malloc(sizeof(Neuron *) * layers);
    for (unsigned int i = 0; i < layers; i++) {
        network->layers[i] = (Neuron *)malloc(sizeof(Neuron) * neurons[i]);
        neuron_init(network->layers[i]);
    }

    network->layer_count = layers;
    network->synapses = NULL;
    network->synapse_count = 0;
    network->neuron_count = (unsigned int *)malloc(sizeof(unsigned int) * layers);
    memcpy(network->neuron_count, neurons, sizeof(unsigned int) * layers);

    return network;
}

/* Free all memory allocated for the neural network */
void destroy_neural_network(NN *network) {
    for (unsigned int i = 0; i < network->layer_count; i++) {
        for (unsigned int j = 0; j < network->neuron_count[i]; j++) {
            neuron_destroy(&network->layers[i][j]);
        }
        free(network->layers[i]);
    }
    free(network->layers);
    free(network->neuron_count);
    for (unsigned int i = 0; i < network->synapse_count; i++) {
        free(network->synapses[i]);
    }
    free(network->synapses);
    free(network);
}

/* Generates a synapse between the given neurons */
void add_synapse(NN *network, Neuron *input, Neuron *output) {
    assert(network);
    assert(input);
    assert(output);

    // create synapse
    Synapse *synapse = (Synapse *)malloc(sizeof(Synapse));
    synapse->input = input;
    synapse->output = output;
    synapse->weight = (float)rand() / (float)RAND_MAX;

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
void generate_synapses(NN *network) {
    assert(network);

    for (unsigned int layer = 1; layer < network->layer_count; layer++) {
        for (unsigned int neuron1 = 0; neuron1 < network->neuron_count[layer - 1]; neuron1++) {
            for (unsigned int neuron2 = 0; neuron2 < network->neuron_count[layer]; neuron2++) {
                add_synapse(network, &network->layers[layer - 1][neuron1], &network->layers[layer][neuron2]);
            }
        }
    }
}

/* Sets an input neuron to a given value */
void set_input(NN *network, unsigned int index, float value) {
    assert(network);
    assert(index < network->neuron_count[0]);
    network->layers[0][index].input = value;
}

/* Reads the current value of the given output neuron */
float read_output(NN *network, unsigned int index) {
    assert(network);
    assert(index < network->layer_count);
    return network->layers[network->layer_count - 1][index].output;
}

/* Recalculate the neural network and set the out put neurons dependent on the
    states of the input neurons */
void calculate(NN *network) {
    assert(network);

    for (unsigned int layer = 0; layer < network->layer_count; layer++) {
        //~ printf("\n--- layer %d ---\n", layer);
        for (unsigned int neuron = 0; neuron < network->neuron_count[layer]; neuron++) {
            neuron_fire(&network->layers[layer][neuron]);
            //~ printf(":: input: %f   output: %f\n", network->layers[layer][neuron].input, network->layers[layer][neuron].output);
        }
    }
}

/* Returns the current error factor of the neural network */
float error_factor(NN *network, TD *train_data) {
    assert(network);
    assert(train_data);

    float error = 0.0;
    for (unsigned int test = 0; test < train_data->data_count; test++) {
        // set inputs
        for (unsigned int i = 0; i < train_data->input_count; i++) {
            network->layers[0][i].input = train_data->input[test][i];
        }
        calculate(network);
        // calculate error in output
        for (unsigned int i = 0; i < train_data->output_count; i++) {
            error += fabs(train_data->output[test][i] - network->layers[network->layer_count - 1][i].output);
        }
    }
    return error / (train_data->output_count * train_data->data_count);
}

/* Backpropagates the network hidden layers recursively */
void backpropagate_hidden(NN *network, float *previous_deltas, float learning_factor, float momentum, unsigned int layer) {
    if (layer == 0) {
        return;
    }

    Neuron *neuron;
    float error, change;
    float *deltas = (float *)malloc(sizeof(float) * network->neuron_count[layer]);
    for (unsigned int current = 0; current < network->neuron_count[layer]; current++) {
        neuron = &network->layers[layer][current];
        error = 0.0;
        for (unsigned int previous = 0; previous < network->neuron_count[layer + 1]; previous++) {
            error += previous_deltas[previous] * neuron->output;
        }
        deltas[current] = error * neuron_dsigmoid(neuron);
        for (unsigned int j = 0; j < neuron->input_count; j++) {
            change = neuron->inputs[j]->input->output * deltas[current];
            neuron->inputs[j]->weight += (change * learning_factor) + (neuron->last_change * momentum);
            neuron->last_change = change;
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

    Neuron *neuron;
    float error, change;
    float *deltas;
    for (unsigned int test = 0; test < train_data->data_count; test++) {
        // set inputs
        for (unsigned int i = 0; i < train_data->input_count; i++) {
            network->layers[0][i].input = train_data->input[test][i];
        }
        calculate(network);
        // backpropagate
        deltas = (float *)malloc(sizeof(float) * network->neuron_count[layer]);
        for (unsigned int i = 0; i < network->neuron_count[layer]; i++) {
            neuron = &network->layers[layer][i];
            error = train_data->output[test][i] - neuron->output;
            deltas[i] = error * neuron_dsigmoid(neuron);
            for (unsigned int j = 0; j < neuron->input_count; j++) {
                change = neuron->inputs[j]->input->output * deltas[i];
                neuron->inputs[j]->weight += (change * learning_factor) + (neuron->last_change * momentum);
                neuron->last_change = change;
            }
        }
        // recurse
        backpropagate_hidden(network, deltas, learning_factor, momentum, layer - 1);
        free(deltas);
    }
}

/* train the neural network with the defined data */
float train(NN *network, TD *train_data, float learning_factor, float momentum) {
    assert(network);
    assert(train_data);

    float error = error_factor(network, train_data);
    backpropagate_output(network, train_data, learning_factor, momentum);
    return error_factor(network, train_data) - error;
}