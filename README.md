Source code for, AutoRec, an autoencoder based model for collaborative filtering. This package also includes implementation of
RBM based collaborative filtering model(RBM-CF).


Dependencies
============
* cython
* progressbar
* envoy
* climin


Configuration
=============
Models are defined in yaml configuration file. Configuration file consists of three sections
* **data**:
	In this section, we define data sources and model save path
	- **train** : path of the training file
	- **test** : path of the test file
	- **save** : path for saving the model
* **param**:
	In this section, we define network training parameters
	- **lamda**: list of regularization paramter per each layer
	- **max_iter**: maximum number of iteration
	- **batch_size**: size of the batch
	- **optimizer**: Choice of the optimizer (lbfgs, rprop, rmsprop)
	- **reg_bias**:  whether to regularize bias or not
	- **beta**: sparsity control parameter
	- **num_threads**: maximum number of threads to be used while doing some of the matrix operations (set it to number of CPU cores)
* **layer**:
	In this section, we define the network architecture. Layers are defined by layer index(starting from 1).
	Note that, layer index should be defined in ascending order (For eg: 1, 2, 3).
	Each layer is defined as 
	- Layer index:
		+ **activation**: Type of activation function (identity, sigmoid, relu, nrelu, tanh)
		+ **num_nodes**: number of nodes in the given layer
		+ **type**: layer type (input, hidden, output)
		+ **partial**: whether the data in the given layer is partially observed or not (applicable only to input/output nodes)
		+ **binary** : whether to enforce binary coding in the layer or not

Installation/Running
====================

First, you will need to build the cython modules. Build cython modules by running
* bash buildCython.sh 

Running Autorec model
* cd nn/autorec
* PYTHONPATH=\<NNRec_PATH\> python learner.py -c \<CONF_PATH\>

Running RBMCF model
* cd nn/cfrbm
* PYTHONPATH=\<NNRec_PATH\> python learner.py -c \<CONF_PATH\>




