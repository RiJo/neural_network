#neural_network
This project contains some different implementations of neural networks.

###Author(s)
- [Rikard Johansson](https://github.com/RiJo)

####Supervised Learning
**Progress: 100%**
- Multilayer perceptron (MLP)
	- This is a supervised learning neural network. 
	- It contains several layers of neurones where all layers are connected by synapses.

####Reinforcement Learning
**Progress: 0%**
- Markov decision process (MDP)
	- This is a reinforcement learning neural network.

##Roadmap
####Todo:
- [ ] Fix `nn_connected()` according to comment.
- [ ] Implement `nn_fully_connected()`
- [ ] Rename output/input of neurone. (confusion with the same values of synapse)

####Ideas:
- [ ] Make it possible to create/remove synapses/neurones in a network.
- [ ] Support for zero (`0`) hidden layers, input directly connected to output layer.
- [ ] Support for synapses within same layer? 
- [ ] Support for synapses to previous layer?
- [ ] Make network dynamic: when learning it should add/remove synapses if necessary.
- [ ] Fix `nn_generate_synapses()` to not add 100% synapses (related to dynamic synapses), rename to something else?
