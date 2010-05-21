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

float input[NEURON_COUNT_INPUT];
float output[NEURON_COUNT_OUTPUT];
unsigned int neurons[] = {NEURON_COUNT_INPUT, NEURON_COUNT_HIDDEN, NEURON_COUNT_OUTPUT};

int main(int argc, char **argv) {
    // Train data
    TD *data = train_data_create(NEURON_COUNT_INPUT, NEURON_COUNT_OUTPUT);
    assert(data);
    assert(data->input_count == NEURON_COUNT_INPUT);
    assert(data->output_count == NEURON_COUNT_OUTPUT);

    input[0] = 0.75;
    input[1] = 0.25;
    output[0] = -0.25;
    output[1] = 0.75;
    train_data_add(data, input, output);
    assert(data->data_count == 1);

    input[0] = 0.25;
    input[1] = 0.75;
    output[0] = 1.0;
    output[1] = 0.0;
    train_data_add(data, input, output);
    assert(data->data_count == 2);

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
        (void)nn_train(network, data, learning_factor, momentum);
        if (i % 10000 == 0) {
            float current_error = nn_error_factor(network, data);
            assert(current_error < previous_error);
            printf("    error: %.2f%%\n", current_error * 100);
            previous_error = current_error;
        }
    }
    printf("\n");

    // Dump structure to file
    FILE *file = fopen(DUMP_FILE, "w");
    assert(file);
    nn_dump_to_file(network, file);
    fclose(file);
    
    /*

    printf("Neural network dumped to file \"%s\"\n\n", DUMP_FILE);

    printf("\n################################################################################\n");
    printf("Loading neural network from file \"%s\"\n\n", argv[1]);

    // test load network from file
    FILE *file = fopen (argv[1], "r");
    if (file == NULL) {
        fprintf(stderr, "Error: could not open file \"%s\" for reading", argv[1]);
        exit(EXIT_FAILURE);
    }
    network = nn_load_from_file(file);
    fclose(file);
    if (network == NULL) {
        fprintf(stderr, "Error: could not load neural network from file");
        exit(EXIT_FAILURE);
    }

    printf("Neural network loaded: \"%s\"\n", network->comment);

    // show results of input 1
    nn_set_input(network, 0, train_data->input[0][0]);
    nn_set_input(network, 1, train_data->input[0][1]);
    nn_calculate(network);
    printf("\ninput  1:   %.2f \t%.2f\n", train_data->input[0][0], train_data->input[0][1]);
    printf("output 1:   %.2f \t%.2f\n", train_data->output[0][0], train_data->output[0][1]);
    printf("real   1:   %.2f \t%.2f\n\n", nn_read_output(network, 0), nn_read_output(network, 1));

    // show results of input 2
    nn_set_input(network, 0, train_data->input[1][0]);
    nn_set_input(network, 1, train_data->input[1][1]);
    nn_calculate(network);
    printf("input  2:   %.2f \t%.2f\n", train_data->input[1][0], train_data->input[1][1]);
    printf("output 2:   %.2f \t%.2f\n", train_data->output[1][0], train_data->output[1][1]);
    printf("real   2:   %.2f \t%.2f\n\n", nn_read_output(network, 0), nn_read_output(network, 1));

*/
    nn_destroy(network);
    train_data_destroy(data);

    printf("ALL TESTS PASSED\n");

    return 0;
}
