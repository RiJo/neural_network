#neural_network
This project is a neural network implemented in C using the [Supervised Learning](http://en.wikipedia.org/wiki/Supervised_learning) pattern.

###Author(s)
- [Rikard Johansson](https://github.com/RiJo)

####Further Information
**Progress: 100%**
- Uses Multilayer perceptron (MLP) technology.
	- It contains several layers of neurones where all layers are connected by synapses.
- Built to make learning and comprehension of foreign data types easier.
- Features
	- Train neurone network(s) using backward propagation. (only output layer)
	- Dump neurone network(s) into a textfile.
	- Load neurone network(s) from a textfile.
	- Comment field to make distinction between network dumps.

>This project was originally developed for a university project where the purpose was to identity a coffee cup in an image captured by a mounted camera on a moving robot. We ran the image through some filters to get the contours which were converted into a polygon and passed to the neural network. I wanted to train the network on my computer to make loading time more efficient on an embedded device. (which is the reason for a simple dump and load textile)

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
- [ ] Markov decision process (MDP)
