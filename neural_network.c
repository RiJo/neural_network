#include "neural_network.h"

// forward declaration of private functions
int nn_path_between(Neuron *, Neuron *);

/* Initialize the neural network with all neurons */
NN *nn_create(unsigned int layers, unsigned int *neurons) {
#ifdef DEBUG
    assert(layers > 0);
    assert(neurons);
#endif

    // allocate memory
    NN *network = (NN *)malloc(sizeof(NN));
    if (network == NULL) {
        return NULL;
    }

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
#ifdef DEBUG
    assert(network);
#endif

    for (unsigned int layer = 0; layer < network->layer_count; layer++) {
        for (unsigned int neuron = 0; neuron < network->neuron_count[layer]; neuron++) {
            neuron_destroy(&(network->layers[layer][neuron]));
        }
        free(network->layers[layer]); // free neuron block
        network->layers[layer] = NULL;
    }
    free(network->layers);
    network->layers = NULL;
    free(network->neuron_count);
    network->neuron_count = NULL;
    for (unsigned int synapse = 0; synapse < network->synapse_count; synapse++) {
        synapse_destroy(network->synapses[synapse]);
    }
    free(network->synapses);
    network->synapses = NULL;

    free(network);
}

size_t nn_size(NN *network) {
#ifdef DEBUG
    assert(network);
#endif

    return (
        sizeof(NN) +
        (sizeof(Synapse) * network->synapse_count) +
        (sizeof(Neuron) * (network->neuron_count[0] + network->neuron_count[1]+network->neuron_count[2]))
    );
}

/* Initalize a neural network based upon the data in a prestored file (see the
    header file of this module for more info) */
NN *nn_load_from_file(FILE *file) {
#ifdef DEBUG
    assert(file);
#endif

    // parse header
    char structure[255];
    memset(structure, '\0', 255);
    if (fscanf (file, "[NN-" NN_FILE_DUMP_VERSION "<%[0123456789:]>]\r\n", structure) != 1) {
        fprintf(stderr, "Error: invalid version of neural network dump file. Expected version: %s\n", NN_FILE_DUMP_VERSION);
        return NULL;
    }

    unsigned int layer_count = 0;
    for (unsigned int i = 0; i < strlen(structure); i++) {
        if (structure[i] == ':') layer_count++;
    }
    if (layer_count == 0) {
        fprintf(stderr, "Error: could not find any layers in neural network dump file\n");
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
    if (network == NULL) {
        return NULL;
    }
    network->layer_count = layer_count;
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
#ifdef DEBUG
    assert(network);
    assert(file);
#endif

    if (network->layer_count == 0) {
        fprintf(stderr, "Error: could not find any layers in neural network\n");
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
#ifdef DEBUG
    assert(network);
    assert(input_layer < network->layer_count);
    assert(output_layer < network->layer_count);
    assert(input_neuron < network->neuron_count[input_layer]);
    assert(output_neuron < network->neuron_count[output_layer]);
#endif

    Neuron *input = &network->layers[input_layer][input_neuron];
    Neuron *output = &network->layers[output_layer][output_neuron];
    Synapse *synapse = synapse_create(input, output, pow(output_layer, output_layer) / 5.0);

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
#ifdef DEBUG
    assert(network);
    //assert(network->synapse_count == 0);
#endif

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
#ifdef DEBUG
    assert(network);
    assert(index < network->neuron_count[0]);
#endif

    network->layers[0][index].input = value;
}

/* Reads the current value of the given output neuron */
float nn_read_output(NN *network, unsigned int index) {
#ifdef DEBUG
    assert(network);
    assert(index < network->layer_count);
#endif

    return network->layers[network->layer_count - 1][index].output;
}

/* Recalculate the neural network and set the out put neurons dependent on the
    states of the input neurons */
void nn_calculate(NN *network) {
#ifdef DEBUG
    assert(network);
#endif

    for (unsigned int layer = 0; layer < network->layer_count; layer++) {
        for (unsigned int neuron = 0; neuron < network->neuron_count[layer]; neuron++) {
            neuron_fire(&network->layers[layer][neuron]);
        }
    }
}

/* checks weather there are synapses between all input neurons and the output
   neurons */
/* TODO: implement check for output->input */
int nn_connected(NN *network) {
#ifdef DEBUG
    assert(network);
#endif

    int connected;
    
    // Check if all inputs are connected to any output
    for (unsigned int input = 0; input < network->neuron_count[0]; input++) {
        connected = 0;
        for (unsigned int output = 0; output < network->neuron_count[network->layer_count - 1]; output++) {
            connected = nn_path_between(network->layers[0] + input, network->layers[network->layer_count - 1] + output);
            if (connected)
                break;
        }
        if (!connected)
            return 0;
    }
    return 1;
}

/* checks weather there are a path (synapses) between the given neurons */
/* TODO: implement check whole graph structure */
int nn_path_between(Neuron *source, Neuron *destination) {
#ifdef DEBUG
    assert(source);
    assert(destination);
#endif

    Synapse *synapse;
    Neuron *neuron;

    for (unsigned int output = 0; output < source->output_count; output++) {
        synapse = source->outputs[output];
        neuron = synapse->output;
        if (neuron == destination || nn_path_between(neuron, destination))
            return 1;
    }
    return 0;
}
