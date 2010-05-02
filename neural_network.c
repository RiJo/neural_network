#include "neural_network.h"

/* Initialize the neural network with all neurons */
NN *create_neural_network(unsigned int layers, unsigned int *neurons) {
    srand (time(NULL));

    // allocate memory
    NN *network = (NN *)malloc(sizeof(NN));
    network->layers = (Neuron **)malloc(sizeof(Neuron *) * layers);
    for (unsigned int i = 0; i < layers; i++) {
        network->layers[i] = (Neuron *)malloc(sizeof(Neuron) * neurons[i]);
    }

    network->layer_count = layers;
    network->synapse_count = 0;
    network->neuron_count = (unsigned int *)malloc(sizeof(unsigned int) * layers);
    memcpy(network->neuron_count, neurons, sizeof(unsigned int) * layers);

    return network;
}

/* Free all memory allocated for the neural network */
void destroy_neural_network(NN *network) {
    for (unsigned int i = 0; i < network->layer_count; i++) {
        free(network->layers[i]);
    }
    free(network->layers);
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

    // bind input neuron
    input->count.outputs++;
    input->outputs = realloc(input->outputs, sizeof(Synapse *) * input->count.outputs);
    input->outputs[input->count.outputs - 1] = synapse;

    // bind output neuron
    output->count.inputs++;
    output->inputs = realloc(output->inputs, sizeof(Synapse *) * output->count.inputs);
    output->inputs[output->count.inputs - 1] = synapse;

    network->synapse_count++;
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

/* Backpropagates the network and returns the error factor delta */
void backpropagate(NN *network, TD *train_data, float learning_factor, unsigned int layer) {
    Neuron *neuron;
    for (unsigned int test = 0; test < train_data->data_count; test++) {
        // set inputs
        for (unsigned int i = 0; i < train_data->input_count; i++) {
            network->layers[0][i].input = train_data->input[test][i];
        }
        calculate(network);
        // backpropagate output
        for (unsigned int i = 0; i < network->neuron_count[layer]; i++) {
            neuron = &network->layers[layer][i];
            float diff = train_data->output[test][i] - neuron->output;
            for (unsigned int j = 0; j < neuron->count.inputs; j++) {
                neuron->inputs[i]->weight += (diff * learning_factor);
            }
        }
    }
}


float train(NN *network, TD *train_data, float learning_factor) {
    float error = error_factor(network, train_data);
    backpropagate(network, train_data, learning_factor, network->layer_count - 1);
    return error_factor(network, train_data) - error;
}