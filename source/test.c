#include "neural_network.h"
#include "supervised.h"

#include <stdio.h>

#define DUMP_FILE "test.dmp"

// forward declaration
TD *generate_train_data(void);

TD *generate_train_data(void) {
    unsigned int inputs = 2;
    unsigned int outputs = 2;

    TD *data = td_create(inputs, outputs);

    float input[inputs];
    float output[outputs];

    input[0] = 0.75;
    input[1] = 0.25;
    output[0] = -0.25;
    output[1] = 0.75;
    td_add(data, input, output);

    input[0] = 0.25;
    input[1] = 0.75;
    output[0] = 1.0;
    output[1] = 0.0;
    td_add(data, input, output);

    return data;
}

void show_results(NN *network, TD* train_data) {
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
}

int main(int argc, char **argv) {

    TD *train_data = generate_train_data();
    NN *network = NULL;

    if (argc == 1) {
        printf("\n################################################################################\n");
        printf("Creating new neural network...\n");

        unsigned int neurons[] = {2, 5, 2};
        network = nn_create(3, neurons);

        printf("Network inputs: %d   hidden: %d   outputs: %d\n\n", network->neuron_count[0], network->neuron_count[1], network->neuron_count[2]);
        
        printf("Network is connected: %d\n", nn_connected(network));
        printf("Generating synapses...\n");
        nn_generate_synapses(network);
        printf("Network synapses: %d\n\n", network->synapse_count);

        printf("Size of network: %d bytes\n\n", (int)nn_size(network));

        printf("Results before training:");
        show_results(network, train_data);

        float error = 0.0;
        for (int i = 0; i < 100000; i++) {
            error += nn_train(network, train_data);
            if (i % 10000 == 0) {
                printf("   err: %.5f - %.5f\t\t output 2: %.2f   %.2f\n",
                        error, nn_error_factor(network, train_data), nn_read_output(network, 0), nn_read_output(network, 1));
                error = 0.0;
            }
        }

        printf("\nResults after training:");
        show_results(network, train_data);

        FILE *fp = fopen(DUMP_FILE, "w");
        nn_dump_to_file(network, fp);
        fclose(fp);
        
        printf("Neural network dumped to file \"%s\"\n\n", DUMP_FILE);
    }
    else {
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

        printf("Neural network loaded\n");

        printf("\nResults of loaded network:");
        show_results(network, train_data);
    }

    nn_destroy(network);
    td_destroy(train_data);

    return EXIT_SUCCESS;
}
