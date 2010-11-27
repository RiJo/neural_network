/*
 * This module ...
 */

#ifndef _SUPERVISED_H_
#define _SUPERVISED_H_

#include "neural_network.h"
#include "train_data.h"

float nn_error_factor(NN *, TD *);
float nn_train(NN *, TD *, float, float);

#endif
