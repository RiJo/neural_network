#include "neural_network.h"

#include <stdio.h>

#define DUMP_FILE "test.dmp"
#define COMMENT "this is a test comment"

#define LAYER_COUNT 3
#define NEURON_COUNT_INPUT 2
#define NEURON_COUNT_HIDDEN 3
#define NEURON_COUNT_OUTPUT 2
#define SYNAPSE_COUNT ((NEURON_COUNT_INPUT * NEURON_COUNT_HIDDEN) + (NEURON_COUNT_HIDDEN * NEURON_COUNT_OUTPUT))

#define TRAIN_ITERATIONS 100000
#define ERROR_TOLERANCE 0.00001
#define OUTPUT_TOLERANCE 0.00001

float input[NEURON_COUNT_INPUT];
float output[NEURON_COUNT_OUTPUT];
unsigned int neurons[] = {NEURON_COUNT_INPUT, NEURON_COUNT_HIDDEN, NEURON_COUNT_OUTPUT};

int main(int argc, char **argv) {
    // Train data
    TD *train_data = train_data_create(NEURON_COUNT_INPUT, NEURON_COUNT_OUTPUT);
    assert(train_data);
    assert(train_data->input_count == NEURON_COUNT_INPUT);
    assert(train_data->output_count == NEURON_COUNT_OUTPUT);

    input[0] = 0.75;
    input[1] = 0.25;
    output[0] = -0.25;
    output[1] = 0.75;
    train_data_add(train_data, input, output);
    assert(train_data->data_count == 1);

    input[0] = 0.25;
    input[1] = 0.75;
    output[0] = 1.0;
    output[1] = 0.0;
    train_data_add(train_data, input, output);
    assert(train_data->data_count == 2);

    // Create new neural network
    NN *network = nn_create(3, neurons);
    assert(network);

    nn_generate_synapses(network);
    assert(network->synapse_count == SYNAPSE_COUNT);

    nn_set_comment(network, COMMENT);
    assert(network->comment);

    // Train neural network with train data
    float learning_factor = 0.5;
    float momentum = 0.1;
    printf("\n  Training...\n");
    float previous_error = 1.0;
    for (int i = 0; i < TRAIN_ITERATIONS; i++) {
        (void)nn_train(network, train_data, learning_factor, momentum);
        if (i % 10000 == 0) {
            float current_error = nn_error_factor(network, train_data);
            assert(current_error < previous_error);
            printf("    error: %.2f%%\n", current_error * 100);
            previous_error = current_error;
        }
    }
    printf("\n");

    // Dump structure to file
    FILE *dump = fopen(DUMP_FILE, "w");
    assert(dump);
    nn_dump_to_file(network, dump);
    fclose(dump);

    // Load network structure from file
    FILE *load = fopen (DUMP_FILE, "r");
    assert(load);
    NN *loaded = nn_load_from_file(load);
    fclose(load);
    assert(loaded);

    // Assure data
    assert(nn_size(network) == nn_size(loaded));
    assert(network->layer_count == loaded->layer_count);
    assert(network->synapse_count == loaded->synapse_count);
    for (int i = 0; i < network->layer_count; i++) {
        assert(network->neuron_count[i] == loaded->neuron_count[i]);
    }
    assert(strcmp(network->comment, loaded->comment) == 0);

    assert(
        nn_error_factor(loaded, train_data) > nn_error_factor(network, train_data) - ERROR_TOLERANCE &&
        nn_error_factor(loaded, train_data) < nn_error_factor(network, train_data) + ERROR_TOLERANCE
    );
    for (int data = 0; data < train_data->data_count; data++) {
        for (int input = 0; input < train_data->input_count; input++) {
            nn_set_input(network, input, *train_data->input[input]);
            nn_set_input(loaded, input, *train_data->input[input]);
        }
        nn_calculate(network);
        nn_calculate(loaded);
        for (int output = 0; output < train_data->output_count; output++) {
            assert(
                nn_read_output(loaded, output) > nn_read_output(network, output) - OUTPUT_TOLERANCE &&
                nn_read_output(loaded, output) < nn_read_output(network, output) + OUTPUT_TOLERANCE
            );
        }
    }

    nn_destroy(network);
    train_data_destroy(train_data);

    printf("ALL TESTS PASSED\n");

    return 0;
}
