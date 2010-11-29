#include "supervised.h"

// forward declaration of private functions
void backpropagate_output(NN *, TD *);
void backpropagate_hidden(NN *, float *, unsigned int);

/* Returns the current error factor of the neural network */
float nn_error_factor(NN *network, TD *train_data) {
#ifdef DEBUG
    assert(network);
    assert(train_data);
#endif

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
#ifdef DEBUG
    assert(network);
    assert(previous_deltas);
#endif

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
#ifdef DEBUG
    assert(network);
    assert(train_data);
#endif

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
#ifdef DEBUG
    assert(network);
    assert(train_data);
    assert(train_data->input_count == network->neuron_count[0]);
    assert(train_data->output_count == network->neuron_count[network->layer_count - 1]);
    assert(nn_connected(network));
#endif

    float error = nn_error_factor(network, train_data);
    backpropagate_output(network, train_data);
    return nn_error_factor(network, train_data) - error;
}
