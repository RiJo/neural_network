#include "neural_network.h"

// forward declaration of private functions
void backpropagate_output(NN *, TD *, float , float);
void backpropagate_hidden(NN *, float *, float , float , unsigned int);

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

    network->comment = NULL;
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
    free(network->comment);
    free(network);
}

size_t nn_size(NN *network) {
    return (
        sizeof(NN) +
        (sizeof(Synapse) * network->synapse_count) +
        (sizeof(Neuron) * (network->neuron_count[0] + network->neuron_count[1]+network->neuron_count[2])) +
        ((network->comment != NULL) ? sizeof(char) * strlen(network->comment) : 0)
    );
}

/* Initalize a neural network based upon the data in a prestored file (see the
    header file of this module for more info) */
NN *nn_load_from_file(FILE *file) {
    assert(file);

    // parse header
    char structure[255];
    char comment[512];
    memset(structure, '\0', 255);
    memset(comment, '\0', 512);
    if (fscanf (file, "[NN-" NN_FILE_DUMP_VERSION "<%[0123456789:]>]\r\n", structure) != 1) {
        printf("Error: invalid version of neural network dump file. Expected version: %s\n", NN_FILE_DUMP_VERSION);
        return NULL;
    }
    if (fgets(comment, 512, file) == NULL) {
        printf("Error: Invalid header, could not parse comment.\n");
        return NULL;
    }
    comment[strlen(comment) - 1] = '\0';
    comment[strlen(comment) - 1] = '\0';

    unsigned int layer_count = 0;
    for (unsigned int i = 0; i < strlen(structure); i++) {
        if (structure[i] == ':') layer_count++;
    }
    if (layer_count == 0) {
        printf("Error: Could not find any layers in neural network dump file\n");
        return NULL;
    }
    layer_count++;
    unsigned int neuron_count[layer_count];
    char *offset = structure;
    for (unsigned int i = 0; i < layer_count; i++) {
        neuron_count[i] = atoi(offset);
        offset = strchr(offset, ':') + 1;
    }

    // create network
    NN *network = nn_create(layer_count, neuron_count);
    network->comment = (char *)malloc(sizeof(char) * (strlen(comment) + 1));
    strcpy(network->comment, comment);
    network->comment[strlen(comment)] = '\0';
    network->layer_count = layer_count;
    network->neuron_count = (unsigned int *)malloc(sizeof(unsigned int) * layer_count);
    memcpy(network->neuron_count, neuron_count, sizeof(unsigned int) * layer_count);
    unsigned int layer1, layer2, neuron1, neuron2;
    float weight, change;
    Synapse *synapse;
    while(fscanf(file, "%d:%d:%d:%d:%f:%f\r\n", &layer1, &neuron1, &layer2, &neuron2, &weight, &change) == 6) {
        nn_add_synapse(network, layer1, neuron1, layer2, neuron2);
        synapse = network->synapses[network->synapse_count - 1];
        synapse->weight = weight;
        synapse->change = change;
    }

    return network;
}

/* Dump the neural networks data into a file for later use (see the header file
    of this module for more info) */
void nn_dump_to_file(NN *network, FILE *file) {
    assert(network);
    assert(file);

    if (network->layer_count == 0) {
        printf("Error: Could not find any layers in neural network\n");
        return;
    }

    // write header
    char structure[255];
    memset(structure, '\0', 255);

    sprintf(structure, "%d", network->neuron_count[0]);
    for (unsigned int i = 1; i < network->layer_count; i++) {
        sprintf(&structure[strlen(structure)], ":%d", network->neuron_count[i]);
    }
    fprintf(file, "[NN-%s<%s>]\r\n", NN_FILE_DUMP_VERSION, structure);
    if (network->comment != NULL) {
        fprintf(file, "%s\r\n", network->comment);
    }
    else {
        fprintf(file, "(no comment)\r\n");
    }

    // write data
    Synapse *synapse;
    unsigned int layer1, layer2, neuron1, neuron2;
    layer1 = layer2 = neuron1 = neuron2 = 0;
    for (unsigned int i = 0; i < network->synapse_count; i++) {
        synapse = network->synapses[i];
        for (unsigned int layer = 0; layer < network->layer_count; layer++) {
            for (unsigned int neuron = 0; neuron < network->neuron_count[layer]; neuron++) {
                if (&network->layers[layer][neuron] == synapse->input) {
                    layer1 = layer;
                    neuron1 = neuron;
                }
                if (&network->layers[layer][neuron] == synapse->output) {
                    layer2 = layer;
                    neuron2 = neuron;
                }
            }
        }
        fprintf(file, "%d:%d:%d:%d:%f:%f\r\n", layer1, neuron1, layer2, neuron2, synapse->weight, synapse->change);
    }
}

/* Generates a synapse between the given neurons */
void nn_add_synapse(NN *network, unsigned int input_layer, unsigned int input_neuron,
        unsigned int output_layer, unsigned int output_neuron) {
    assert(network);
    assert(input_layer < network->layer_count);
    assert(output_layer < network->layer_count);
    assert(input_neuron < network->neuron_count[input_layer]);
    assert(output_neuron < network->neuron_count[output_layer]);

    Neuron *input = &network->layers[input_layer][input_neuron];
    Neuron *output = &network->layers[output_layer][output_neuron];
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
                nn_add_synapse(network, layer - 1, neuron1, layer, neuron2);
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
    float test_error;
    for (unsigned int test = 0; test < train_data->data_count; test++) {
        // set inputs
        for (unsigned int input = 0; input < train_data->input_count; input++) {
            network->layers[0][input].input = train_data->input[test][input];
        }
        nn_calculate(network);
        // calculate error in outputs
        test_error = 0.0;
        for (unsigned int output = 0; output < train_data->output_count; output++) {
            test_error += fabs(train_data->output[test][output] - network->layers[network->layer_count - 1][output].output);
        }
        error += (test_error / (float)train_data->output_count);
    }
    return error / (float)train_data->data_count;
}

/* Backpropagates the network hidden layers recursively */
void backpropagate_hidden(NN *network, float *previous_deltas, float learning_factor, float momentum, unsigned int layer) {
    if (layer == 0) {
        return; // base case
    }

    Neuron *current_neuron;
    Synapse *synapse;
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
            synapse = current_neuron->inputs[input];
            synapse->weight += (change * learning_factor) + (synapse->change * momentum);
            synapse->change = change;
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
    Synapse *synapse;
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
                synapse = current_neuron->inputs[input];
                synapse->weight += (change * learning_factor) + (synapse->change * momentum);
                synapse->change = change;
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
