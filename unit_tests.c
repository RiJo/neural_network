#include "neural_network.h"
#include "supervised.h"

#include <stdio.h>

#define DUMP_FILE "test.dmp"

#define LAYER_COUNT 3
#define NEURON_COUNT_INPUT 2
#define NEURON_COUNT_HIDDEN 3
#define NEURON_COUNT_OUTPUT 2
#define SYNAPSE_COUNT ((NEURON_COUNT_INPUT * NEURON_COUNT_HIDDEN) + (NEURON_COUNT_HIDDEN * NEURON_COUNT_OUTPUT))

#define TRAIN_ITERATIONS 100000
#define ERROR_TOLERANCE 0.00001
#define OUTPUT_TOLERANCE 0.00001

unsigned int current_test_number = 1;

float input[NEURON_COUNT_INPUT];
float output[NEURON_COUNT_OUTPUT];
unsigned int neurons[] = {NEURON_COUNT_INPUT, NEURON_COUNT_HIDDEN, NEURON_COUNT_OUTPUT};

TD *train_data;
NN *network;

void test_announce(char *comment) {
    printf(" Test %2i: %s\n", current_test_number++, comment);
}

void connectivity_test() {
    test_announce("connectivity test");
    unsigned int neurons[] = {2, 1, 2};
    network = nn_create(3, neurons);

    // no synapses
    assert(!nn_connected(network));

    // synapses from input 1 to output 1
    nn_add_synapse(network, 0, 0, 1, 0);
    nn_add_synapse(network, 1, 0, 2, 0);
    assert(!nn_connected(network));

    // synapses from input 1 to all outputs
    nn_add_synapse(network, 1, 0, 2, 1);
    assert(!nn_connected(network));

    // synapses from all inputs to all outputs
    nn_add_synapse(network, 0, 1, 1, 0);
    assert(nn_connected(network));

    nn_destroy(network);
}

void mirror_test() {
    test_announce("mirror test through bottle neck");
    train_data = td_create(2, 2);

    unsigned int steps = 5;
    float step_size = 1.0 / steps;
    for (unsigned int i = 0; i <= steps; i++) {
        // Equal
        //~ printf("1: %.2f  2: %.2f\n", step_size * (float)i, step_size * (float)i);
        input[0] = step_size * (float)i;
        input[1] = step_size * (float)i;
        output[0] = step_size * (float)i;
        output[1] = step_size * (float)i;
        td_add(train_data, input, output);

        // 1st dec. 2nd inc.
        //~ printf("1: %.2f  2: %.2f\n", 1.0 - (step_size * (float)i), step_size * (float)i);
        input[0] = 1.0 - (step_size * (float)i);
        input[1] = step_size * (float)i;
        output[0] = 1.0 - (step_size * (float)i);
        output[1] = step_size * (float)i;
        td_add(train_data, input, output);

        // 1st inc. 2nd dec.
        //~ printf("1: %.2f  2: %.2f\n", step_size * (float)i, 1.0 - (step_size * (float)i));
        input[0] = step_size * (float)i;
        input[1] = 1.0 - (step_size * (float)i);
        output[0] = step_size * (float)i;
        output[1] = 1.0 - (step_size * (float)i);
        td_add(train_data, input, output);
    }

    unsigned int neurons[] = {2, 1, 2};
    network = nn_create(3, neurons);
    nn_generate_synapses(network);
    nn_generate_synapses(network);

    // Train neural network with train data
    printf("\n  Training...\n");
    float previous_error = 1.0;
    for (unsigned int i = 0; i < TRAIN_ITERATIONS; i++) {
        (void)nn_train(network, train_data);
        if (i % 10000 == 0) {
            float current_error = nn_error_factor(network, train_data);
            assert(current_error < previous_error);
            printf("    error: %f%%\n", current_error * 100);
            previous_error = current_error;
        }
    }
    printf("\n");
}

int main(int argc, char **argv) {
    // Train data
    train_data = td_create(NEURON_COUNT_INPUT, NEURON_COUNT_OUTPUT);
    
    test_announce("return value of td_create()");
    assert(train_data);

    test_announce("train data input count");
    assert(train_data->input_count == NEURON_COUNT_INPUT);

    test_announce("train data output count");
    assert(train_data->output_count == NEURON_COUNT_OUTPUT);

    input[0] = 0.75;
    input[1] = 0.25;
    output[0] = -0.25;
    output[1] = 0.75;
    td_add(train_data, input, output);
    test_announce("train data one data item");
    assert(train_data->data_count == 1);

    input[0] = 0.25;
    input[1] = 0.75;
    output[0] = 1.0;
    output[1] = 0.0;
    td_add(train_data, input, output);
    test_announce("train data several data items");
    assert(train_data->data_count == 2);

    // Create new neural network
    network = nn_create(3, neurons);
    test_announce("return value of nn_create()");
    assert(network);

    nn_generate_synapses(network);
    test_announce("network synapse count");
    assert(network->synapse_count == SYNAPSE_COUNT);

    // Train neural network with train data
    test_announce("learning rate");
    float previous_error = 1.0;
    for (unsigned int i = 0; i < TRAIN_ITERATIONS; i++) {
        (void)nn_train(network, train_data);
        float current_error = nn_error_factor(network, train_data);
        assert(current_error < previous_error + 0.0001);
        if (i % 10000 == 0) {
            printf("    error: %.2f%%\n", current_error * 100);
            
        }
        previous_error = current_error;
    }

    // Dump structure to file
    test_announce("dump network to file");
    FILE *dump = fopen(DUMP_FILE, "w");
    assert(dump);
    nn_dump_to_file(network, dump);
    fclose(dump);

    // Load network structure from file
    test_announce("load network from file");
    FILE *load = fopen (DUMP_FILE, "r");
    assert(load);
    NN *loaded = nn_load_from_file(load);
    fclose(load);
    assert(loaded);

    // Confirm data
    test_announce("confirm data in loaded network");
    assert(nn_size(network) == nn_size(loaded));
    assert(network->layer_count == loaded->layer_count);
    assert(network->synapse_count == loaded->synapse_count);
    for (unsigned int i = 0; i < network->layer_count; i++) {
        assert(network->neuron_count[i] == loaded->neuron_count[i]);
    }

    assert(
        nn_error_factor(loaded, train_data) > nn_error_factor(network, train_data) - ERROR_TOLERANCE &&
        nn_error_factor(loaded, train_data) < nn_error_factor(network, train_data) + ERROR_TOLERANCE
    );
    for (unsigned int data = 0; data < train_data->data_count; data++) {
        for (unsigned int input = 0; input < train_data->input_count; input++) {
            nn_set_input(network, input, *train_data->input[input]);
            nn_set_input(loaded, input, *train_data->input[input]);
        }
        nn_calculate(network);
        nn_calculate(loaded);
        for (unsigned int output = 0; output < train_data->output_count; output++) {
            assert(
                nn_read_output(loaded, output) > nn_read_output(network, output) - OUTPUT_TOLERANCE &&
                nn_read_output(loaded, output) < nn_read_output(network, output) + OUTPUT_TOLERANCE
            );
        }
    }

    nn_destroy(network);
    td_destroy(train_data);

    connectivity_test();

    //~ mirror_test();

    printf("ALL TESTS PASSED\n");

    return 0;
}
