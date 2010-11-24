#include "train_data.h"

TD *train_data_create(unsigned int inputs, unsigned int outputs) {
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

void train_data_destroy(TD *data) {
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

void train_data_add(TD *data, float *input, float *output) {
    assert(data);
    assert(input);
    assert(output);

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
