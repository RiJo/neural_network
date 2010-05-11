#include "neural_network.h"

#include <stdio.h>

// forward declaration
TD *generate_train_data(void);

TD *generate_train_data(void) {
    unsigned int inputs = 2;
    unsigned int outputs = 2;

    TD *data = train_data_create(inputs, outputs);

    float input[inputs];
    float output[outputs];

    input[0] = 0.75;
    input[1] = 0.25;
    output[0] = -0.25;
    output[1] = 0.75;
    train_data_add(data, input, output);

    input[0] = 0.25;
    input[1] = 0.75;
    output[0] = 1.0;
    output[1] = 0.0;
    train_data_add(data, input, output);

    return data;
}

int main() {
    unsigned int neurons[] = {2, 5, 2};
    NN *network = nn_create(3, neurons);
    nn_generate_synapses(network);
    TD *train_data = generate_train_data();

    printf("\nNetwork inputs: %d   hidden: %d   outputs: %d\n", network->neuron_count[0], network->neuron_count[1], network->neuron_count[2]);
    printf("Network synapses: %d\n", network->synapse_count);

    // show results of input 1
    nn_set_input(network, 0, train_data->input[0][0]);
    nn_set_input(network, 1, train_data->input[0][1]);
    nn_calculate(network);
    printf("input  1:   %.2f \t%.2f\n", train_data->input[0][0], train_data->input[0][1]);
    printf("output 1:   %.2f \t%.2f\n", train_data->output[0][0], train_data->output[0][1]);
    printf("real   1:   %.2f \t%.2f\n\n", nn_read_output(network, 0), nn_read_output(network, 1));

    // show results of input 2
    nn_set_input(network, 0, train_data->input[1][0]);
    nn_set_input(network, 1, train_data->input[1][1]);
    nn_calculate(network);
    printf("input  2:   %.2f \t%.2f\n", train_data->input[1][0], train_data->input[1][1]);
    printf("output 2:   %.2f \t%.2f\n", train_data->output[1][0], train_data->output[1][1]);
    printf("real   2:   %.2f \t%.2f\n\n", nn_read_output(network, 0), nn_read_output(network, 1));

    /*float learning_factor = 0.5;
    float momentum = 0.1;
    float error = 0.0;
    for (int i = 0; i < 100000; i++) {
        error += nn_train(network, train_data, learning_factor, momentum);
        if (i % 10000 == 0) {
            printf("   err: %.5f - %.5f\t\t output 2: %.2f   %.2f\n",
                    error, nn_error_factor(network, train_data), nn_read_output(network, 0), nn_read_output(network, 1));
            error = 0.0;
        }
    }*/

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

    FILE *fp = fopen("test.dmp", "w");
    nn_dump_to_file(network, fp, "this is a comment");
    fclose(fp);

    nn_destroy(network);

    printf("########################################################################\n");
    printf(" Network loaded from file...\n\n");

    // test load network from file
    FILE *file = fopen ("test.dmp","r");
    NN *loaded = nn_load_from_file(file);
    fclose(file);
    
    printf("Loaded neural network: \"%s\"\n", loaded->comment);

    // show results of input 1
    nn_set_input(loaded, 0, train_data->input[0][0]);
    nn_set_input(loaded, 1, train_data->input[0][1]);
    nn_calculate(loaded);
    printf("\ninput  1:   %.2f \t%.2f\n", train_data->input[0][0], train_data->input[0][1]);
    printf("output 1:   %.2f \t%.2f\n", train_data->output[0][0], train_data->output[0][1]);
    printf("real   1:   %.2f \t%.2f\n\n", nn_read_output(loaded, 0), nn_read_output(loaded, 1));

    // show results of input 2
    nn_set_input(loaded, 0, train_data->input[1][0]);
    nn_set_input(loaded, 1, train_data->input[1][1]);
    nn_calculate(loaded);
    printf("input  2:   %.2f \t%.2f\n", train_data->input[1][0], train_data->input[1][1]);
    printf("output 2:   %.2f \t%.2f\n", train_data->output[1][0], train_data->output[1][1]);
    printf("real   2:   %.2f \t%.2f\n\n", nn_read_output(loaded, 0), nn_read_output(loaded, 1));

    nn_destroy(loaded);

    train_data_destroy(train_data);

    return 0;
}
