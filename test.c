#include "neural_network.h"

#include <stdio.h>

TD *generate_train_data() {
    unsigned int inputs = 2;
    unsigned int outputs = 2;

    TD *data = create_train_data(inputs, outputs);

    float input[inputs];
    float output[outputs];

    input[0] = 0.75;
    input[1] = 0.25;
    output[0] = 0.25;
    output[1] = 0.75;
    add_train_data(data, input, output);

    //~ input[0] = 0.75;
    //~ input[1] = 0.25;
    //~ output[0] = 0.25;
    //~ output[1] = 0.75;
    //~ add_train_data(data, input, output);

    input[0] = 0.25;
    input[1] = 0.75;
    output[0] = 1.0;
    output[1] = 0.0;
    add_train_data(data, input, output);

    return data;
}

int main() {
    unsigned int neurons[] = {2, 3, 2};
    NN *network = create_neural_network(3, neurons);
    generate_synapses(network);
    TD *train_data = generate_train_data();

    printf("\nNetwork inputs: %d   hidden: %d   outputs: %d\n", network->neuron_count[0], network->neuron_count[1], network->neuron_count[2]);
    printf("Network synapses: %d\n", network->synapse_count);

    // show results of input 1
    set_input(network, 0, train_data->input[0][0]);
    set_input(network, 1, train_data->input[0][1]);
    calculate(network);
    printf("input  1:   %.2f \t%.2f\n", train_data->input[0][0], train_data->input[0][1]);
    printf("output 1:   %.2f \t%.2f\n", train_data->output[0][0], train_data->output[0][1]);
    printf("real   1:   %.2f \t%.2f\n\n", read_output(network, 0), read_output(network, 1));

    // show results of input 2
    set_input(network, 0, train_data->input[1][0]);
    set_input(network, 1, train_data->input[1][1]);
    calculate(network);
    printf("input  2:   %.2f \t%.2f\n", train_data->input[1][0], train_data->input[1][1]);
    printf("output 2:   %.2f \t%.2f\n", train_data->output[1][0], train_data->output[1][1]);
    printf("real   2:   %.2f \t%.2f\n\n", read_output(network, 0), read_output(network, 1));

    float learning_factor = 0.5;
    float momentum = 0.1;
    float error;
    for (int i = 0; i < 1000; i++) {
        error = train(network, train_data, learning_factor, momentum);
        if (i % 100 == 0) {
            printf("   err: %.5f - %.5f\t\t output 2: %.2f   %.2f\n",
                    error, error_factor(network, train_data), read_output(network, 0), read_output(network, 1));
        }
    }

    // show results of input 1
    set_input(network, 0, train_data->input[0][0]);
    set_input(network, 1, train_data->input[0][1]);
    calculate(network);
    printf("\ninput  1:   %.2f \t%.2f\n", train_data->input[0][0], train_data->input[0][1]);
    printf("output 1:   %.2f \t%.2f\n", train_data->output[0][0], train_data->output[0][1]);
    printf("real   1:   %.2f \t%.2f\n\n", read_output(network, 0), read_output(network, 1));

    // show results of input 2
    set_input(network, 0, train_data->input[1][0]);
    set_input(network, 1, train_data->input[1][1]);
    calculate(network);
    printf("input  2:   %.2f \t%.2f\n", train_data->input[1][0], train_data->input[1][1]);
    printf("output 2:   %.2f \t%.2f\n", train_data->output[1][0], train_data->output[1][1]);
    printf("real   2:   %.2f \t%.2f\n\n", read_output(network, 0), read_output(network, 1));

    destroy_train_data(train_data);
    destroy_neural_network(network);
    return 0;
}