#neural_network
This project contains some different implementations of neural networks.

####Author(s)
- [Rikard Johansson](https://github.com/RiJo) 

####Supervised Learning
- Multilayer perceptron (MLP)		|		Progress: 100%
	- This is a supervised learning neural network. It contains several layers of neurons where all layers are connected by synapses.

####Reinforcement Learning
- Markov decision process (MDP)		|		Progress: 0%
	- This is a reinforcement learning neural network.

##Roadmap
####Todo:
- [ ] Fix `nn_connected()` according to comment.
- [ ] implement `nn_fully_connected()`.
- [ ] rename output/input of neuron (confusion with the same values of synapse)

####Ideas:
- [ ] Make it possible to create/remove synapses/neurons in a network
- [ ] Support for 0 (zero) hidden layers, input directly connected to output layer.
- [ ] Support for synapses within same layer? Support for synapses to previous layer?
- [ ] Make network dynamic: when learning it should add/remove synapses if necessary.
- [ ] Fix `nn_generate_synapses()` to not add 100% synapses (related to dynamic synapses), rename to something else?