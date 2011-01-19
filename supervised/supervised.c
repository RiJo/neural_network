#include "supervised.h"

// forward declaration of private functions
void backpropagate_output(NN *, TD *);
void backpropagate_hidden(NN *, float *, unsigned int);

/* Returns the current error factor of the neural network */
float nn_error_factor(NN *network, TD *train_data) {
    ASSERT(network);
    ASSERT(train_data);

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
void backpropagate_hidden(NN *network, float *previous_deltas, unsigned int layer) {
    ASSERT(network);
    ASSERT(previous_deltas);

    if (layer == 0) {
        return; // base case
    }

    Neuron *current_neuron;
    Synapse *synapse;
    float error, change;

    // backpropagate
    float *deltas = (float *)malloc(sizeof(float) * network->neuron_count[layer]);
    for (unsigned int neuron = 0; neuron < network->neuron_count[layer]; neuron++) {
        current_neuron = &network->layers[layer][neuron];
        error = 0.0;
        for (unsigned int output = 0; output > current_neuron->output_count; output++) {
            for (unsigned int previous = 0; previous < network->neuron_count[layer + 1]; previous++) { // shouldn't this loop be only connecting neurons? N<-S->N
                if (current_neuron->outputs[output]->output == network->layers[previous])
                    error += (previous_deltas[previous] * current_neuron->outputs[output]->weight);
            }
        }
        deltas[neuron] = error * neuron_dsigmoid(current_neuron);
    }

    // recurse
    backpropagate_hidden(network, deltas, layer - 1);

    // update values
    for (unsigned int neuron = 0; neuron < network->neuron_count[layer]; neuron++) {
        current_neuron = &network->layers[layer][neuron];
        for (unsigned int input = 0; input < current_neuron->input_count; input++) {
            synapse = current_neuron->inputs[input];
            change = synapse->input->output * deltas[neuron];
            synapse_change(synapse, change);
        }
    }
    free(deltas);
}

/* Backpropagates the network outputs */
void backpropagate_output(NN *network, TD *train_data) {
    ASSERT(network);
    ASSERT(train_data);

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
        }

        // recurse
        backpropagate_hidden(network, deltas, layer - 1);

        // update values
        for (unsigned int neuron = 0; neuron < network->neuron_count[layer]; neuron++) {
            current_neuron = &network->layers[layer][neuron];
            for (unsigned int input = 0; input < current_neuron->input_count; input++) {
                synapse = current_neuron->inputs[input];
                change = synapse->input->output * deltas[neuron];
                synapse_change(synapse, change);
            }
        }
        free(deltas);
    }
}

/* train the neural network with the defined data */
float nn_train(NN *network, TD *train_data) {
    ASSERT(network);
    ASSERT(train_data);
    ASSERT(train_data->input_count == network->neuron_count[0]);
    ASSERT(train_data->output_count == network->neuron_count[network->layer_count - 1]);
    ASSERT(nn_connected(network));

    float error = nn_error_factor(network, train_data);
    backpropagate_output(network, train_data);
    return nn_error_factor(network, train_data) - error;
}

TD *td_create(unsigned int inputs, unsigned int outputs) {
    TD *data = (TD *)malloc(sizeof(TD));
    if (data == NULL) {
        return NULL;
    }
    data->input = NULL;
    data->output = NULL;
    data->data_count = 0;
    data->input_count = inputs;
    data->output_count = outputs;
    return data;
}

void td_destroy(TD *data) {
    ASSERT(data);

    unsigned int i;
    for (i = 0; i < data->input_count; i++) {
        free(data->input[i]);
        data->input[i] = NULL;
    }
    free(data->input);
    data->input = NULL;
    for (i = 0; i < data->output_count; i++) {
        free(data->output[i]);
        data->output[i] = NULL;
    }
    free(data->output);
    data->output = NULL;
    free(data);
}

void td_add(TD *data, float *input, float *output) {
    ASSERT(data);
    ASSERT(input);
    ASSERT(output);

    data->data_count++;

    // set input data
    data->input = (float **)realloc(data->input, sizeof(float *) * data->data_count);
    if (data->input == NULL) {
        fprintf(stderr, "Error: could not allocate memory for train data input values\n");
        return;
    }
    data->input[data->data_count - 1] = (float *)malloc(sizeof(float) * data->input_count);
    memcpy(data->input[data->data_count - 1], input, sizeof(float) * data->input_count);

    // set output data
    data->output = (float **)realloc(data->output, sizeof(float *) * data->data_count);
    if (data->output == NULL) {
        fprintf(stderr, "Error: could not allocate memory for train data output values\n");
        return;
    }
    data->output[data->data_count - 1] = (float *)malloc(sizeof(float) * data->output_count);
    memcpy(data->output[data->data_count - 1], output, sizeof(float) * data->output_count);
}